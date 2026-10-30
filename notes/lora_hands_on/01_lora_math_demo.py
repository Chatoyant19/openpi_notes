"""手把手看 LoRA 的数学: 直接 import openpi 真实的 lora.Einsum,
用很小的 toy shape 跑一遍,把 w / lora_a / lora_b 和前向结果都打印出来。

跑法:
    cd /local/mnt2/workspace/danwa/project/VLA/openpi
    uv run python notes/lora_hands_on/01_lora_math_demo.py
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import lora

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. 造一个和 Gemma attention 里 q_einsum 同构, 但维度很小的 LoRA Einsum
#    真实代码里 shape=(num_heads, width, head_dim), eqn="BTD,NDH->BTNH"
#    这里缩小成 (N=2 个头, D=4 模型宽度, H=3 head_dim), rank=2
# ---------------------------------------------------------------------------
section("1. 构造一个 LoRA Einsum (和 Gemma 的 q_einsum 形状逻辑完全一样,只是缩小了)")

lora_cfg = lora.LoRAConfig(rank=2, alpha=2.0, init_fn=nn.initializers.normal(stddev=0.5))
print("LoRAConfig:", lora_cfg)
print("scaling_value = alpha / rank =", lora_cfg.scaling_value)

eqn = "BTD,NDH->BTNH"
module = lora.Einsum(shape=(2, 4, 3), lora_config=lora_cfg)  # (N, D, H)

rng = jax.random.PRNGKey(0)
x = jax.random.normal(jax.random.PRNGKey(1), (1, 5, 4))  # (B=1, T=5, D=4)

variables = module.init(rng, eqn, x)
params = variables["params"]

print("\n参数树里实际有哪些 key, 形状分别是:")
for k, v in params.items():
    print(f"  {k:10s} shape={v.shape}")

w, w_a, w_b = params["w"], params["lora_a"], params["lora_b"]
print("\nw (基础权重, 形状 (N,D,H)=(2,4,3)):\n", np.array(w))
print("\nlora_a (形状 (N,D,rank)=(2,4,2), 把 D 轴降到 rank):\n", np.array(w_a))
print("\nlora_b (形状 (N,rank,H)=(2,2,3), 把 rank 轴升回 H):\n", np.array(w_b))

# ---------------------------------------------------------------------------
# 2. 看 _make_lora_eqns 到底把 "BTD,NDH->BTNH" 重写成了什么
# ---------------------------------------------------------------------------
section("2. _make_lora_eqns 把原始 einsum 方程拆成了两段")

eqn_a, eqn_b = module._make_lora_eqns(eqn)
print(f"原始 eqn:  {eqn}")
print(f"第一段 eqn_a (x @ lora_a, 把 D 轴换成 L=rank 轴): {eqn_a}")
print(f"第二段 eqn_b (结果 @ lora_b, 把 L 轴换回 H 轴):   {eqn_b}")

# ---------------------------------------------------------------------------
# 3. 手动按公式重新算一遍, 跟 module.apply 的结果对比, 验证公式
#    result = einsum(eqn, x, w) + scaling_value * einsum(eqn_b, einsum(eqn_a, x, w_a), w_b)
# ---------------------------------------------------------------------------
section("3. 手动按公式重算一遍, 和真实 module.apply() 的结果做数值对比")

base = jnp.einsum(eqn, x, w)
lora_mid = jnp.einsum(eqn_a, x, w_a)  # x 先降到 rank 维
lora_out = jnp.einsum(eqn_b, lora_mid, w_b)  # 再从 rank 维升回去
manual_result = base + lora_cfg.scaling_value * lora_out

real_result = module.apply(variables, eqn, x)

print("base (不含 LoRA 的原始投影结果) shape:", base.shape)
print("lora_mid (x 降到 rank=2 维之后的中间结果) shape:", lora_mid.shape)
print("lora_out (再升回 H 维的低秩增量) shape:", lora_out.shape)
print("\n手动公式计算结果 == module.apply() 真实结果 ?",
      bool(jnp.allclose(manual_result, real_result, atol=1e-5)))

# ---------------------------------------------------------------------------
# 4. 证明 LoRA 就是"基础结果 + 一个小增量", 把基础部分单独跑出来对比
# ---------------------------------------------------------------------------
section("4. 把 LoRA 关掉 (lora_config=None), 只跑基础权重, 验证 LoRA 只是加了一个增量")

base_only_module = lora.Einsum(shape=(2, 4, 3), lora_config=None)
base_only_result = base_only_module.apply({"params": {"w": w}}, eqn, x)

delta = real_result - base_only_result
print("只用基础权重 w 算出的结果 == 上面手动算的 base ?",
      bool(jnp.allclose(base_only_result, base, atol=1e-5)))
print("真实结果 - 纯基础结果 (= LoRA 增量) 的范数:", float(jnp.linalg.norm(delta)))
print("这个增量 == scaling_value * lora_out ?",
      bool(jnp.allclose(delta, lora_cfg.scaling_value * lora_out, atol=1e-5)))

# ---------------------------------------------------------------------------
# 5. 看梯度: 如果不做任何冻结, loss 对 w / lora_a / lora_b 是不是都有梯度
#    (这一步是为了说明: LoRA 模块本身并不会自动冻结 w, 冻结是训练循环里
#     另一层的 trainable_filter 做的, 见下一个脚本)
# ---------------------------------------------------------------------------
section("5. 不加任何冻结过滤器时, loss 对 w / lora_a / lora_b 的梯度范数")


def loss_fn(params):
    out = module.apply({"params": params}, eqn, x)
    return jnp.sum(out**2)


grads = jax.grad(loss_fn)(params)
for k, g in grads.items():
    print(f"  grad[{k:10s}] norm = {float(jnp.linalg.norm(g)):.6f}")

print(
    "\n可以看到 w / lora_a / lora_b 三者都有非零梯度 —— LoRA 模块自己并不会冻结基础权重,"
    "\n真正在训练里『只更新 lora_a/lora_b, 冻结 w』靠的是外层的 freeze_filter / trainable_filter,"
    "\n下一个脚本 (02_real_model_param_tree.py) 会在真实的 Pi0 模型上验证这一点。"
)
