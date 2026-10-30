"""在真实的 Pi0 模型上验证 freeze_filter / trainable_filter 到底筛出了哪些参数。

用 nnx.eval_shape 只构造"抽象模型"(只有 shape/dtype, 不实际分配内存/做随机数计算),
这样秒级就能看到完整参数树, 不需要下载任何 checkpoint, 也不用等真正的权重初始化。

跑法:
    cd /local/mnt2/workspace/danwa/project/VLA/openpi
    JAX_PLATFORMS=cpu uv run python notes/lora_hands_on/02_real_model_param_tree.py
"""

import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def describe(config: _pi0_config.Pi0Config, label: str) -> None:
    section(f"{label}: paligemma_variant={config.paligemma_variant}, action_expert_variant={config.action_expert_variant}")

    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    freeze_filter = config.get_freeze_filter()

    all_params = nnx.state(abstract_model, nnx.Param).flat_state()
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))).flat_state()

    def count(flat_state) -> int:
        return sum(v.value.size for v in flat_state.values())

    total_n = count(all_params)
    frozen_n = count(frozen)
    trainable_n = count(trainable)

    print(f"参数路径总数: {len(all_params)}  (frozen {len(frozen)} + trainable {len(trainable)})")
    print(f"参数元素总数: {total_n:,}")
    print(f"  其中冻结 (frozen):    {frozen_n:,}  ({100 * frozen_n / total_n:.2f}%)")
    print(f"  其中可训练 (trainable): {trainable_n:,}  ({100 * trainable_n / total_n:.2f}%)")

    if frozen:
        print("\n被冻结的参数路径示例 (最多显示 8 个, 可以看到全部带 'llm', 不带 'lora'):")
        for path in list(frozen.keys())[:8]:
            v = frozen[(*path,)]
            print(f"  {'/'.join(str(p) for p in path):55s} shape={v.value.shape}")

    print("\n可训练的参数路径 (按前缀分组统计, 看它们到底落在模型哪些部分):")
    groups: dict[str, list[tuple[str, int]]] = {}
    for path, v in trainable.items():
        prefix = path[0] if not isinstance(path[0], int) else "/".join(str(p) for p in path[:2])
        groups.setdefault(str(path[0]), []).append(("/".join(str(p) for p in path), v.value.size))

    for prefix, items in groups.items():
        n = sum(sz for _, sz in items)
        print(f"  顶层模块 {prefix!r:20s}: {len(items):3d} 个参数路径, 共 {n:,} 个元素")

    lora_paths = [p for p in trainable if any("lora" in str(x) for x in p)]
    non_lora_trainable = [p for p in trainable if not any("lora" in str(x) for x in p)]
    print(f"\n  -> 其中路径名里带 'lora' 的: {len(lora_paths)} 个")
    print(f"  -> 不带 'lora' 但仍然可训练的 (说明它们本来就不在 freeze_filter 的 '.*llm.*' 范围内): "
          f"{len(non_lora_trainable)} 个")
    if non_lora_trainable:
        print("     举例:")
        for path in non_lora_trainable[:6]:
            v = trainable[path]
            print(f"       {'/'.join(str(p) for p in path):40s} shape={v.value.shape}")


# ---------------------------------------------------------------------------
# A. 全量微调 (baseline): 两个专家都不是 lora variant, freeze_filter 应该是空的
# ---------------------------------------------------------------------------
describe(_pi0_config.Pi0Config(), "A. 全量微调 (pi0_libero 用的默认配置)")

# ---------------------------------------------------------------------------
# B. 只对 VLM 专家 (gemma_2b) 开 LoRA, action expert 仍然全量微调
# ---------------------------------------------------------------------------
describe(
    _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora"),
    "B. 只给 VLM 专家开 LoRA, action expert 仍全量微调",
)

# ---------------------------------------------------------------------------
# C. 两个专家都开 LoRA —— 这正是 pi0_libero_low_mem_finetune 实际用的配置
# ---------------------------------------------------------------------------
describe(
    _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
    "C. 两个专家都开 LoRA (= pi0_libero_low_mem_finetune 的真实配置)",
)
