"""把训练好的 policy 以 WebSocket 服务的形式跑起来，供远端机器人客户端调用。

======================================================================
这个脚本在整个系统里的位置
======================================================================
openpi 把"推理"和"机器人控制"拆成了两个进程，中间用 WebSocket 连起来：

    [机器人侧 / 仿真环境]                    [GPU 服务器]
    examples/<env>/main.py       ──obs──>    scripts/serve_policy.py  ← 本文件
    WebsocketClientPolicy        <─actions─   WebsocketPolicyServer

这么拆的理由很实际：机器人上那台机器通常没有能跑 3B 模型的显卡，而且
ALOHA/DROID 这些平台的驱动往往绑定在 ROS + 老 Python 版本上，和 JAX 的依赖
冲突。拆开之后，机器人侧只需要装一个极轻的 `openpi-client` 包（见
packages/openpi-client，纯 numpy + websockets，没有 JAX/torch）。
背景说明见 notes/remote_inference.md。

本文件自己的逻辑很薄，真正干活的是三层：
  1. `policy_config.create_trained_policy()` —— 加载权重 + 拼装数据变换流水线
  2. `policy.Policy` —— 单次推理：输入变换 → 模型采样 → 输出反变换
  3. `websocket_policy_server.WebsocketPolicyServer` —— 网络层，收发 msgpack

======================================================================
跑法
======================================================================
    # 用某个环境的默认 checkpoint（会自动从 GCS 下载并缓存）
    uv run scripts/serve_policy.py --env=LIBERO

    # 用自己训练出来的 checkpoint（注意 policy:checkpoint 这个子命令写法）
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_libero_low_mem_finetune \
        --policy.dir=checkpoints/pi0_libero_low_mem_finetune/my_run/10000

    # 录下每一次推理的输入输出，用来排查"模型在真机上行为不对"的问题
    uv run scripts/serve_policy.py --env=LIBERO --record

======================================================================
几个容易踩的点
======================================================================
- **norm_stats 来自 checkpoint，不是来自配置的 assets 目录**。
  `create_trained_policy` 读的是 `<checkpoint>/assets/<asset_id>`，这样才能保证
  推理时的归一化参数和当初训练时**完全一致**。如果你训练后改过数据集、重跑了
  `compute_norm_stats.py`，配置目录里的统计量会变，但 checkpoint 里的不会——
  这是故意的（policy_config.py:59-64）。
- `--env` 只在"不指定 checkpoint"时才起作用。一旦走了 `policy:checkpoint`
  子命令，env 就被忽略了，实际用哪套数据变换完全由 `--policy.config` 决定。
- 服务监听 `0.0.0.0`，也就是**对整个网络开放、且没有任何认证**。
  只在可信内网里用，别直接暴露到公网。
- 模型是在**第一次推理时**才真正编译（JAX jit）的，所以第一帧会特别慢
  （几十秒量级），之后才稳定。客户端侧要注意别把第一帧的耗时当成常态。
"""

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    # 这里的"环境"指机器人平台 / benchmark，决定了用哪套默认 checkpoint。
    # 真正影响数据变换的是对应配置里的 data transforms，不是这个枚举本身。
    ALOHA = "aloha"  # 真机 ALOHA 双臂
    ALOHA_SIM = "aloha_sim"  # ALOHA 的 MuJoCo 仿真
    DROID = "droid"  # DROID 单臂数据集/平台
    LIBERO = "libero"  # LIBERO 仿真 benchmark


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    # 必须和训练时用的配置名一致：模型结构、数据变换、asset_id 全从这里取。
    # 填错了不一定报错，但会静默地用错一套 transform，表现为"动作看着合理但就是不对"。
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    # 支持本地路径，也支持 gs:// 开头的 GCS 路径（由 download.maybe_download 处理，
    # 会缓存到 ~/.cache/openpi，见 src/openpi/shared/download.py）。
    # 注意要指到**具体的 step 目录**（如 .../10000），不是 exp 根目录。
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""

    # 空 dataclass，纯粹作为 tyro 的"另一个分支"存在（见下面 policy 字段的 union 类型）。


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    # 只有在 policy=Default() 时才有意义；走 policy:checkpoint 子命令时这个值被忽略。
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    # 兜底的语言指令。π0 是 VLA（视觉-语言-动作）模型，每次推理都需要一条文本指令；
    # 如果客户端发来的 obs 里没有 "prompt" 键，就用这里的值填上
    # （实现是 transforms.InjectDefaultPrompt，见 policy_config.py:79）。
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    # 开启后每一步的输入输出都会存成 policy_records/step_N.npy，用于离线复盘。
    # 代价是每步都写盘，会拖慢推理，别在正式评测时开。
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    # 这是个 tagged union：tyro 会把它映射成 `policy:checkpoint` / `policy:default`
    # 两个子命令。所以命令行里要写成
    #     policy:checkpoint --policy.config=... --policy.dir=...
    # 而不是 --policy=...
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
# 官方预训练权重，都放在公开的 GCS bucket 上，第一次用会自动下载并缓存。
# 体积不小（π0 约 3B 参数），首次启动要等一会儿。
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        # 注意这里指的是 pi05_base 而不是 pi05_aloha：真机 ALOHA 没有发布专门微调过的
        # 权重，用的是**通用基座**，靠配置里的 ALOHA 数据变换来适配接口。
        # 也就是说开箱即用的效果有限，真机上基本都需要自己采数据微调。
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """按环境查表，加载官方默认 checkpoint。"""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        # get_config 按名字到 config.py 的 _CONFIGS 列表里查训练配置，
        # 配置决定了模型结构和数据变换；checkpoint.dir 提供权重和 norm_stats。
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据命令行参数造出 policy，这里是"默认权重 / 自训权重"两条路的分叉点。"""
    match args.policy:
        case Checkpoint():
            # 用户显式指定了配置名 + checkpoint 目录，env 参数此时完全不参与。
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    # 1) 加载模型 + 拼装数据变换流水线。
    #    这一步会下载/读取 checkpoint、把权重转成 bfloat16、读 norm_stats，
    #    耗时主要在这里。但注意：此时还**没有**做 JAX 编译，
    #    真正的 jit 编译发生在第一次 infer 时。
    policy = create_policy(args)

    # 2) 先把 metadata 取出来。
    #    必须在下面包 PolicyRecorder 之前取——PolicyRecorder 只代理了 infer()，
    #    并没有转发 .metadata 属性（见 policy.py:113-135），包完就取不到了。
    #    metadata 里装的是机器人侧需要知道的信息，比如 ALOHA 的 reset_pose
    #    （config.py:570），客户端连上来的第一帧就会收到它。
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    # 3) 可选：套一层装饰器，把每步的输入输出落盘到 policy_records/。
    #    这是典型的装饰器模式——对 server 来说它还是个普通 policy。
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 4) 打印本机地址，方便在另一台机器上填客户端的 --host。
    #    这里只是日志，拿到的 IP 不一定是机器人能访问到的那个网卡地址
    #    （多网卡 / 容器环境下经常不准），实际用 `ip addr` 确认更稳。
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 5) 起服务并阻塞在这里。
    #    协议细节见 websocket_policy_server.py：连上后先推一帧 metadata，
    #    之后就是「收 obs → policy.infer() → 回 actions」的循环，
    #    数据用 msgpack 编码（能直接传 numpy 数组），且 compression=None、
    #    max_size=None——图像帧很大，关掉压缩和大小限制是为了压低延迟。
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",  # 监听所有网卡：方便远程连，但也意味着无认证暴露，仅限可信内网
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    # force=True 覆盖掉其他库（JAX / absl 等）可能已经装好的 logging 配置，
    # 否则这里的 level=INFO 可能不生效、看不到上面的日志。
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
