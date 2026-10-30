"""计算某个训练配置所需的「归一化统计量」(normalization statistics)。

======================================================================
这个脚本在整条流水线里的位置
======================================================================
训练之前**必须**先跑一次这个脚本，否则 `scripts/train.py` 会直接报错：
    "Normalization stats not found. Make sure to run
     `scripts/compute_norm_stats.py --config-name=<your-config>`"
（报错源头见 `training/data_loader.py:176-180`）

原因是：真实机器人数据里各个维度的量纲天差地别（比如关节角是弧度 ±3.14，
夹爪开合可能是 0/1，末端位置可能是米或毫米），直接喂给模型，量纲大的维度会
主导梯度、量纲小的维度几乎学不动。所以训练时会在数据流水线里插一个
`transforms.Normalize`，用「每个维度的均值/标准差」或「1%/99% 分位数」
把每一维各自压到可比的范围。而那些均值/标准差/分位数，就是这个脚本扫一遍
数据集统计出来的，存成 json 落盘，训练和推理时再读回来复用。

具体压到什么范围，取决于模型类型（`use_quantile_norm = model_type != PI0`，
`training/config.py:187`），两套公式见 `transforms.py:137-145`：
  - π0        → z-score:  (x - mean) / std
               结果是均值 0、标准差 1，约 68% 落在 [-1,1] 内，**可以超出**；
  - π0.5/FAST → 分位数:   (x - q01) / (q99 - q01) * 2 - 1
               把 1%~99% 区间**精确**映射到 [-1,1]，只有约 2% 的离群点在外面。
这也是为什么 `RunningStats` 要同时统计 mean/std **和** q01/q99——两条路都得支持。

======================================================================
归一化只统计 state 和 actions，而且 actions 是双向的
======================================================================
本脚本只统计 `state` 和 `actions` 两个字段（见 main 里的 `keys`），图像不参与
——图像的归一化是 /255 之类的固定规则，不依赖数据集统计。

这两个字段里，`state` 只是输入，但 `actions` **既是训练时的监督目标、又是推理时
模型的输出**。模型学到的是「预测归一化后的动作」，所以推理出来的数必须先还原回
物理单位才能发给机器人，否则机器人收到的是量纲完全错误的指令。
于是推理时有一对严格镜像的流水线（`policies/policy_config.py:77-89`）：

    inputs:   repack.inputs → data_transforms.inputs → Normalize   → model_transforms.inputs
    outputs:  model_transforms.outputs → Unnormalize → data_transforms.outputs → repack.outputs

`Normalize` 和 `Unnormalize` 处在完全对称的深度上，公式互逆
（`(x-mean)/std` ↔ `x*std+mean`，见 `transforms.py:170-181`）。
换句话说：本脚本算出来的这一份统计量，正向用于喂数据、反向用于还原动作，
两头用的必须是同一份数字——这也是它必须落盘、而不能每次重算的根本原因。

顺带一个容易看懵的细节：`policies/policy.py:92-95` 把明明用不到的 `state`
也塞进了模型输出字典：

    outputs = {"state": inputs["state"], "actions": self._sample_actions(...)}

这是因为 `Unnormalize` 用的是 `strict=True`（`transforms.py:167`），而 strict 模式下
`apply_tree` 会检查 **norm_stats 里的每个 key 都必须出现在数据里**
（`transforms.py:415-418`）。norm_stats 恰好有 `state` 和 `actions` 两个 key，
所以输出里少了 `state` 就会直接抛 ValueError。顺便 `state` 也被还原回了物理单位，
调试时可以直接看。

======================================================================
为什么要单独算一遍、而不是训练时在线算
======================================================================
1) 归一化参数必须在**整个数据集**上保持一致。如果每个 batch 各自算自己的
   均值方差，同一个动作在不同 batch 里会被归一化成不同的数，模型就学不到
   稳定的映射了。
2) 推理/部署时（`serve_policy.py`）只有单帧观测，根本没法统计，必须复用
   训练时存下来的同一份 norm_stats（这也是为什么它会被打包进 checkpoint 的
   `assets/` 目录里，见 `training/config.py:41`）。
3) 这是个纯 I/O 密集的一次性工作，没必要每次训练都重复做。

======================================================================
跑法
======================================================================
    # 完整扫一遍数据集（慢，但统计最准）
    uv run scripts/compute_norm_stats.py --config-name=pi0_libero_low_mem_finetune

    # 只扫大约 2000 帧的随机子集（快，适合 smoke test）
    uv run scripts/compute_norm_stats.py --config-name=pi0_libero_low_mem_finetune --max-frames 2000

    # 配置名必须写成 --config-name=xxx，不能像 train.py 那样当位置参数传。
    # 原因：train.py 用 _config.cli()（config.py:978）把配置注册成子命令；
    # 本脚本用 tyro.cli(main)，参数是从 main 的函数签名推导出来的。

输出：assets/<config.name>/<repo_id>/norm_stats.json
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """把样本字典里所有「字符串类型」的字段丢掉。

    数据集里除了数值张量，还会带上文本字段（最典型的是 `prompt`，也就是
    "pick up the black bowl" 这类任务指令）。在**训练**流水线里这些文本是有用的
    —— 后面的 `model_transforms` 会把它丢给 tokenizer 变成 token id。
    但在**本脚本**里：
      1) 我们只统计 state / actions 两个数值字段，文本对统计量毫无贡献；
      2) 下游的 collate / JAX 数组转换不支持字符串 dtype，留着会直接报错。
    所以这里在流水线末尾加一道过滤，把字符串字段摘干净。

    `np.issubdtype(..., np.str_)` 用来判断一个字段的 dtype 是否属于字符串家族。
    """

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """为「LeRobot 格式数据集」(绝大多数配置走的路径) 建一个只读的 dataloader。

    返回 (dataloader, num_batches)。num_batches 要单独返回，是因为 dataloader
    本身是无限循环的迭代器（见 `data_loader.py:452-468`，数据集跑完会重新开一轮），
    不会自己停，所以必须由调用方拿着这个数字去限制循环次数 / 喂给 tqdm 当进度条总长。
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 建底层数据集。内部会去 HuggingFace 拉取/读缓存，并按 action_horizon 把
    # 未来若干步的动作打包成一个 chunk（delta_timestamps 机制，见
    # `data_loader.py:140-146`）。所以取出来的 actions 形状是
    # (action_horizon, action_dim) 而不是单帧的 (action_dim,)。
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # 【本脚本最关键的一处设计】这里套的变换必须和训练时**完全对齐**。
    #
    # 训练时的完整流水线是（`data_loader.py:183-191`）：
    #     repack_transforms.inputs      <- 改 key 名，把数据集字段映射成 openpi 约定的字段
    #     data_transforms.inputs        <- 机器人相关的处理，如拼接各关节、调整动作维度
    #     Normalize(norm_stats)         <- ★ 归一化就插在这个位置
    #     model_transforms.inputs       <- tokenize 文本、resize 图像等模型侧处理
    #
    # 本脚本只套前两段，也就是恰好停在 Normalize **之前**的那个状态。
    # 这样统计出来的均值/方差，正好是 Normalize 将来要作用的那个坐标空间里的数字。
    # 如果这里少套一层或多套一层，统计量就对不上训练时真正被归一化的张量，
    # 归一化会失效甚至把数据推到更离谱的范围。
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        # 只想快速扫一部分：算出需要多少个 batch，并且**必须打开 shuffle**。
        # 否则只会顺序读到数据集最前面的几条轨迹，那几条可能全是同一个任务、
        # 同一个场景，统计出来的均值方差完全不能代表整个数据集。
        # 打乱之后取前 num_batches 个 batch，等价于在全集上做随机采样。
        # 注意整除会向下取整，实际扫到的帧数是 num_batches * batch_size，
        # 可能略少于 max_frames。
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        # 完整扫一遍：顺序读就行，不需要打乱（反正每条都会被用到）。
        # 同样因为整除，末尾凑不满一个 batch 的尾巴会被丢掉。
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,  # 多进程并行读盘/解码，纯粹为了加速，不影响统计结果
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """为「RLDS / TFDS 格式数据集」建 dataloader —— 目前仅用于 DROID。

    DROID 数据集太大（约两千万帧），没有做成 LeRobot 格式，而是走 TensorFlow
    Datasets 的流式读取，所以单独走一条分支（见 `data_loader.py:154-169`）。
    """
    # 和 torch 路径的区别之一：batching 发生在数据集**内部**（TF 的 pipeline 里），
    # 这里传 batch_size 进去让它自己攒批，拿出来的每个元素已经是一个 batch 了。
    # shuffle=False：统计量不依赖顺序，而且这条路径不做子集采样（见下面的 max_frames 处理），
    # 省掉 shuffle buffer 可以显著降低内存占用、加快启动。
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)

    # 因为上面出来的元素已经是批好的，所以这里必须 is_batched=True：
    # 变换函数都是按「单条样本」写的，`IterableTransformedDataset` 会先把 batch
    # 拆成单条、逐条变换、再重新 stack 回去（见 `data_loader.py:79-91`）。
    # 套的变换列表和 torch 路径完全一致，同样停在 Normalize 之前。
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        # `DroidRldsDataset.__len__` 直接 return 20_000_000（`droid_rlds_dataset.py:245-248`），
        # 是过滤后样本数的粗略估计值——真要遍历一遍数一数代价太大，所以写死了。
        # 这意味着这里算出的 num_batches 也只是个近似值，仅用于 tqdm 进度条和循环上限。
        num_batches = len(dataset) // batch_size

    # 这一层只是个很薄的包装：攒批已经在数据集内部做完了，它只负责数够
    # num_batches 个就停下来（见 `data_loader.py:486-490`）。
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """入口。

    Args:
        config_name: 训练配置名，必须是 `training/config.py` 里 `_CONFIGS` 注册过的，
            比如 "pi0_libero_low_mem_finetune"。统计量是**跟着配置走**的，因为不同配置
            的 repack/data transforms 不同，归一化空间也就不同。
        max_frames: 最多扫多少帧。None 表示扫完整个数据集。调小可以几分钟跑完，
            适合先把链路跑通；正式训练前建议不传，扫全量拿到可靠的统计量。
    """
    # 1. 按名字取出训练配置，再由它派生出数据配置。
    #    `config.data.create(...)` 会顺手尝试加载**已有的** norm_stats
    #    （`config.py:186`），加载失败只会打一条 log 不会报错——这正是本脚本
    #    第一次运行时的情形（文件还不存在）。
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 2. 根据数据格式二选一。只有 DROID 会设置 rlds_data_dir，其余都走 torch 分支。
    #    注意 batch_size 直接复用训练配置里的值：它只影响扫数据的速度和粒度，
    #    不影响最终统计量（均值是按样本数加权累积的，见下面第 4 步）。
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 3. 只统计这两个字段：
    #    - "state":   机器人本体状态（关节角、夹爪等），模型的条件输入之一
    #    - "actions": 要预测的动作序列，形状 (action_horizon, action_dim)
    #    图像不统计——图像的归一化是固定的 /255 之类的规则，不依赖数据集统计。
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # 4. 流式地扫一遍数据，边扫边更新统计量。
    #    用 RunningStats 而不是「全部读进内存再 np.mean」是因为数据集可能有几千万帧，
    #    装不下。它的做法（`shared/normalize.py:30-71`）：
    #      - 均值 / 平方均值：按样本数加权的在线更新，数学上和一次性算全量**完全等价**，
    #        所以 batch_size 取多少都不影响结果；
    #      - 方差 = 平方均值 - 均值², 再开根号得标准差；
    #      - 分位数：维护 5000 个 bin 的直方图近似计算，所以 q01/q99 是**近似值**
    #        （min/max 变大时还会重新分桶，见 `_adjust_histograms`）。
    #    另外 `update()` 里会先 reshape(-1, 最后一维)，意味着 batch 维和
    #    action_horizon 这个时间维都被摊平了——即 chunk 里每一个时间步都算一条独立样本，
    #    统计的是「每个动作维度」的分布。
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    # 5. 收口，算出最终的 mean / std / q01 / q99。
    #    两套参数都存，训练时用哪一套由模型类型决定（见本文件开头的归一化公式一节）。
    #    注意：这行的字典推导式里 `stats` 被重新绑定了，但 `stats.items()` 在
    #    推导式开始前就已求值，所以能正常工作——只是读起来容易绕，别被带偏。
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 6. 落盘成 assets/<config.name>/<repo_id>/norm_stats.json。
    #    repo_id 形如 "physical-intelligence/libero" 自带斜杠，会生成嵌套目录，
    #    `normalize.save` 内部用 mkdir(parents=True) 兜住了（`normalize.py:134-138`）。
    #
    #    ⚠️ 这里有个容易踩的坑：写入用的是 `repo_id`，但训练时**读取**用的是
    #    `asset_id`（= `assets.asset_id or repo_id`，见 `config.py:181,194`）。
    #    绝大多数配置没有自定义 asset_id，两者相同，所以没问题；但像
    #    `AssetsConfig(asset_id="trossen")` 这种配置，本脚本写到 <repo_id>/ 下，
    #    训练却去 <trossen>/ 下找，就会对不上，需要手动把文件挪过去。
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # tyro 会根据 main 的函数签名自动生成 CLI：
    # 位置参数 config_name，可选参数 --max-frames。
    tyro.cli(main)
