"""π0 / π0.5 的训练主入口。

======================================================================
整体结构：三个阶段
======================================================================
1. `main()` 做编排：建 device mesh、建 checkpoint 管理器、建 data loader、
   初始化训练状态，然后进入 for 循环。
2. `init_train_state()` 负责"开局"：造模型、灌入预训练权重、决定哪些参数可训练、
   给可训练参数分配优化器状态、算出 FSDP 分片方案。
3. `train_step()` 是真正的一步训练：前向算 loss → 反向算梯度 → 优化器更新参数。
   它被 `jax.jit` 编译成一个完整的 XLA 计算图（`main` 里的 `ptrain_step`）。

======================================================================
跑法
======================================================================
    # 配置名是位置参数（子命令形式），和 compute_norm_stats.py 的 --config-name= 不同
    uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=my_run --overwrite

    # 任意嵌套字段都能从命令行覆盖（tyro 自动生成）
    uv run scripts/train.py debug --batch-size=64 \
        --lr-schedule.peak-lr=1e-5 --lr-schedule.warmup-steps=0 --log-interval=1

注意：训练前必须先跑 `scripts/compute_norm_stats.py --config-name=<同一个配置>`，
否则 data loader 会因为找不到 norm_stats 直接报错。

======================================================================
显存都花在哪（理解 LoRA 为什么省显存的关键）
======================================================================
一次全量微调，每个可训练参数大致要占 4 份显存：
  参数本身(fp32) + 梯度 + Adam 的一阶动量 m + 二阶动量 v
本脚本有三处设计直接对应这件事：
  - `init_train_state` 里把**冻结**参数转成 bfloat16（第 104 行附近），省一半；
  - `tx.init(...)` 只给**可训练**参数分配优化器状态（第 111 行附近），
    所以 LoRA 下 Adam 的 m/v 只有那 ~14% 参数的量；
  - `train_step` 里用 `nnx.DiffState` 限制求导范围，冻结参数根本不产生梯度。
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    # 把 INFO/WARNING 之类的级别名压成单字母，并带上进程号和源码行号。
    # 这就是日志里 "19:27:14.249 [I] ... (2400202:async_checkpointer.py:276)" 的来源。
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 wandb 实验记录。`--no-wandb-enabled` 可以整个关掉（debug 配置默认就是关的）。"""
    if not enabled:
        # 注意不是直接 return，而是 init 成 disabled 模式——这样后面代码里的
        # `wandb.log(...)` 调用不用加任何 if 判断，全部变成空操作。
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        # 续训时要接回**同一个** wandb run，否则曲线会断成两条。
        # run id 是上次启动时写在 checkpoint 目录里的。
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 全新 run：把整个 TrainConfig 作为超参数快照存进 wandb（事后能查到这次用了什么配置），
        # 并把 run id 落盘，供将来 --resume 用。
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    # params_shape 是"抽象参数树"（只有 shape/dtype，没有真实数值），由下面
    # init_train_state 里的 jax.eval_shape produce 出来。
    loaded_params = loader.load(params_shape)

    # 严格校验：加载回来的权重树必须和模型期望的结构、形状、dtype 完全一致。
    # 这道检查能在训练开始前就拦住"配置和 checkpoint 不匹配"这类错误，
    # 而不是等到跑了几百步才发现结果不对。
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    # weight_loader 对于"checkpoint 里没有、需要新初始化"的参数（最典型的就是 LoRA 的
    # lora_a/lora_b，见 weight_loaders.py 里 missing_regex=".*lora.*"）会原样留下
    # 占位的 ShapeDtypeStruct。这里把它们剔掉，只返回真正有数值的那部分，
    # 剩下的就交给模型自己的随机初始化（见 init() 里 replace_by_pure_dict 的用法）。
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """构造初始 TrainState，并算出它的 FSDP 分片方案。

    返回 (train_state, state_sharding)。`resume=True` 时只返回"抽象"的 state
    （只有 shape/dtype），真实数值稍后由 checkpoint 恢复。
    """
    # 优化器 = 学习率调度 + AdamW。注意 AdamW.create 实际返回的是
    #   optax.chain(optax.clip_by_global_norm(1.0), adamw)   （optimizer.py:85）
    # 梯度裁剪是**包在优化器里**的，这一点在读 train_step 的 grad_norm 时很关键（见那里的注释）。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        # 此时所有参数都是**随机初始化**的（包括 LoRA 的 lora_a/lora_b）。
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            # 把从 checkpoint 加载到的那部分权重覆盖进去。
            # 没被覆盖的参数（如 LoRA adapter）就保留上面的随机初始化值——
            # 这正是"LoRA 从零开始学、基座权重继承预训练"的实现方式。
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)

        # Convert frozen params to bfloat16.
        # 【省显存点 1】冻结的参数永远不会被更新，没必要保持 fp32 精度，
        # 转成 bfloat16 直接省掉一半显存。LoRA 配置下这部分占 ~86% 的参数，
        # 所以这一行的收益非常可观。freeze_filter 的规则见 pi0_config.get_freeze_filter()。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),  # 模型的"结构"部分，和参数分开存（nnx 的 split/merge 模式）
            tx=tx,
            # 【省显存点 2】只给**可训练**参数分配优化器状态。
            # AdamW 每个参数要额外存一阶动量 m 和二阶动量 v（2 倍参数量），
            # 所以 LoRA 下这里只分配 ~14% 参数对应的量，而不是全部。
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            # EMA（指数滑动平均）会额外保存一份参数副本，所以开了 EMA 就要多一份显存。
            # 默认 0.99，但 pi0_libero_low_mem_finetune 显式设成了 None 来省显存（config.py:697）。
            ema_params=None if config.ema_decay is None else params,
        )

    # 关键技巧：jax.eval_shape 只做"形状推演"，不真的执行 init、不分配任何显存。
    # 这样就能在真正建模型之前先拿到完整的参数树结构，用来：
    #   1) 算 FSDP 分片方案（下一行）；
    #   2) 给 weight_loader 做结构校验（下面的 _load_weights_and_validate）。
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 续训：不需要在这里灌权重，直接把"形状模板"和分片方案返回，
        # 由 main() 里的 _checkpoints.restore_state 按这个模板填入真实数值。
        return train_state_shape, state_sharding

    # 全新训练：先把预训练权重加载到主机内存（此时还是 numpy，没上 GPU）。
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    # 用 jit 来执行初始化，好处是：随机初始化 + 权重合并 + 分片 全部在设备上一次完成，
    # 中间结果不会在主机侧堆成一份完整的 fp32 模型副本。
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        # ↑ "捐赠"第 1 个参数(partial_params)的缓冲区：告诉 XLA 这块内存用完即可回收/复用，
        #   不必再留一份。对 3B 参数的模型来说这能省下可观的峰值显存。
        in_shardings=replicated_sharding,  # 输入在每张卡上都是完整副本
        out_shardings=state_sharding,  # 输出按 FSDP 方案切开分散到各卡
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """一步训练：前向 → 反向 → 更新。整个函数会被 jax.jit 编译成一张 XLA 图。

    这是"学习率"和"batch size"真正起作用的地方：
      - batch size 决定 `batch` 里有多少条样本 → 决定 loss 是在多少条数据上取的平均
        → 决定梯度估计的噪声大小；
      - 学习率藏在 `state.tx` 里（由 step 数查调度表得到）→ 决定 `updates` 的幅度。
    """
    # nnx 的 split/merge 模式：参数在 TrainState 里以纯数据形式流转（便于 jit/分片），
    # 用的时候再和 model_def（结构）合并成一个可调用的模型对象。
    model = nnx.merge(state.model_def, state.params)
    model.train()  # 切到训练模式（影响 dropout 等行为）

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # compute_loss 返回的是"每个样本、每个时间步"的 loss（未求平均），
        # 形状大致是 (batch, action_horizon)。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # 这里一次性对 batch 维和时间维取平均 —— 所以 batch_size 越大，
        # 这个平均值的方差越小，梯度估计越稳（这正是之前实验里 batch=64 的
        # grad_norm 波动远小于 batch=2 的原因）。
        return jnp.mean(chunked_loss)

    # 用 step 数去"折"出本步的随机种子：既保证每一步的随机性不同（流匹配的噪声采样、dropout），
    # 又保证整个训练完全可复现（同样的 seed + 同样的 step 必然得到同样的随机数）。
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    # 【省显存点 3】DiffState(0, trainable_filter) = "只对第 0 个参数(model)里
    # 匹配 trainable_filter 的那部分求导"。冻结参数连梯度都不会被计算出来，
    # 既省显存也省算力。trainable_filter 的定义是
    #   nnx.All(nnx.Param, nnx.Not(freeze_filter))    （config.py:550-552）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 取出可训练参数的当前值，和梯度、优化器状态一起交给 optax 算出本步更新量。
    params = state.params.filter(config.trainable_filter)
    # ⚠️ 梯度裁剪发生在这一步**内部**（tx = chain(clip_by_global_norm(1.0), adamw)）。
    #    也就是说下面 info 里报出来的 grad_norm 是**裁剪前的原始范数**，
    #    而真正用于更新的梯度已经被压到全局范数 ≤ 1.0。
    #    这解释了为什么把 peak_lr 调到 1.0 做发散实验时，grad_norm 能飙到上千，
    #    但 param_norm 却是近似线性增长而不是指数爆炸——每步的实际步长是被裁剪兜住的。
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)  # θ ← θ + updates（updates 里已含 -lr 的符号）

    # Update the model in place and return the new full state.
    # 上面只更新了"可训练"的那个子集，这里写回模型，再把**完整**参数树(含冻结部分)取出来，
    # 保证 TrainState.params 始终是完整的一棵树。
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # EMA：维护一份"慢速跟随"的参数副本，通常比瞬时参数更平滑、推理效果更好。
        # 存 checkpoint 时如果有 EMA，导出给推理用的就是 EMA 版本（checkpoints.py:145-152）。
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    # param_norm 只统计"权重矩阵"，刻意排除 bias / scale / 各种 embedding，
    # 并要求 ndim > 1。这样它才是一个干净的"模型权重整体幅度"指标：
    # 突然暴涨说明训练发散，长期一动不动说明更新量相对权重本身小到可以忽略
    # （之前 lr=2.5e-5 的实验里它 20 步都停在 477.8596，就是后者）。
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),  # 裁剪前的原始梯度范数，见上面的警告
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # batch 要能被设备数整除，否则没法均匀切到各张卡上。
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 把 XLA 编译结果缓存到磁盘。train_step 是个很大的图，首次编译可能要几分钟；
    # 有了缓存，第二次用同样配置启动就能跳过编译。
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 全部随机性都从 config.seed 派生 → 同样的配置能复现同样的训练轨迹。
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 设备网格是二维的 (batch, fsdp)，形状为 (device_count // fsdp_devices, fsdp_devices)
    # （sharding.py:17-23）。fsdp_devices=1（默认）时就是纯数据并行；
    # 调大它就把**参数本身**也切开分散到多卡，用于单卡放不下的全量微调。
    mesh = sharding.make_mesh(config.fsdp_devices)
    # 数据沿 batch 和 fsdp 两个轴一起切（DATA_AXIS 是这两者的元组，sharding.py:10）。
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 空 PartitionSpec = 不切，每张卡一份完整副本。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 这里会处理"目录已存在"的三种情形（checkpoints.py:25-36）：
    #   --overwrite → 删掉重建； --resume → resuming=True； 都不给 → 直接报错。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # shuffle=True：每个 epoch 重新打乱样本顺序。
    # 所以并不是"第一步用 1-5 条、第二步用 6-10 条"这样固定顺序循环，
    # 但每个 epoch 内每条样本仍然恰好被用到一次。
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    # 先取一个 batch 出来：既用于打印形状做 sanity check，也作为主循环的第一个输入。
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # 把第一个 batch 的各路摄像头画面横向拼起来传到 wandb。
    # 这是最有效的排错手段之一：图像上下颠倒、通道顺序错、归一化错，肉眼一看就知道。
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # JAX 是异步派发的，上面那行返回时计算可能还没做完。
    # block 一下，确保"初始化耗时"不会被错算到第一步训练上。
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        # 按 init_train_state 返回的形状模板，从 checkpoint 填回真实数值
        # （参数、优化器状态、step 数都会恢复，所以能无缝接着训）。
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 把 train_step 编译成固定的 XLA 图。config 用 partial 固定住（它是静态的 Python 对象，
    # 不能作为 jit 的追踪参数）。
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),  # 捐赠 train_state 的缓冲区：新参数可以直接写在旧参数的内存上，
        #                       省掉一份完整模型副本的峰值显存。
    )

    # 续训时从 checkpoint 里的 step 继续计数，进度条也从那里接上。
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # set_mesh 把 mesh 放进一个全局变量，供模型内部深处的
        # sharding.activation_sharding_constraint 取用（约束中间激活值的分片方式）。
        # 作者在 sharding.py:28-30 解释了为什么用 context manager 而不是层层传参。
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            # 注意：打印出来的不是当前这一步的瞬时值，而是上一个 log_interval 区间内
            # **所有步的平均值**（stack_forest 把 list[dict] 转成 dict[stacked array]，再求均值）。
            # 默认 log_interval=100，所以默认日志里的 loss 是 100 步的平均；
            # 想看逐步的真实波动要加 --log-interval=1。
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []  # 清空，开始累积下一个区间

        # 取下一个 batch 放在训练之后：由于 JAX 异步派发，上面的 ptrain_step 还在 GPU 上跑时，
        # 这里的读盘/解码已经在主机侧并行进行了，相当于免费的数据预取。
        batch = next(data_iter)

        # 存档条件：到了 save_interval 的整数倍（但跳过刚续训的那一步，避免重复存），
        # 或者是最后一步（保证训练结束一定有产出）。
        # 旧 checkpoint 默认只保留最新 1 个，但 step % keep_period == 0 的会被永久保留
        # （checkpoints.py:47-52）。
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # checkpoint 是异步写盘的，主循环结束时可能还有后台任务没完成，必须等一下，
    # 否则进程退出会导致最后一个 checkpoint 损坏或丢失。
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # _config.cli() = tyro.extras.overridable_config_cli(...)（config.py:978-979）：
    # 把 _CONFIGS 里每个配置注册成一个子命令，所以配置名是**位置参数**，
    # 且任意嵌套字段都能用 --a.b.c=value 从命令行覆盖。
    main(_config.cli())
