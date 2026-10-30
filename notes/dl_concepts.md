# 深度学习 / 大模型 / VLA 概念笔记（基于 openpi 代码实证）

> 这份笔记整理的是**概念性知识点**，与 `architecture_analysis.md`（代码结构分析）、
> `norm_stats.md`（上游英文文档）不重复。
>
> 每一条尽量区分两类信息：
> - **通用知识** —— 换个框架、换个项目依然成立；
> - **openpi 实现** —— 这个仓库具体怎么做的，附代码位置。
>
> 凡标注「实测」的数字，都是在本机真实跑出来的，不是估算。

---

## 目录

- [1. LoRA 低秩适配](#1-lora-低秩适配)
- [2. 超参数：学习率与 batch size](#2-超参数学习率与-batch-size)
- [3. 数据归一化](#3-数据归一化)
- [4. 动作表示：绝对 vs 增量](#4-动作表示绝对-vs-增量)
- [5. 训练工程实现](#5-训练工程实现)
- [6. 几个被推翻的直觉](#6-几个被推翻的直觉)
- [7. 配套可跑脚本](#7-配套可跑脚本)

---

## 1. LoRA 低秩适配

### 1.1 核心数学

**通用知识。** 全量微调要学一个和原权重同样大的增量 `ΔW`：

```
y = W·x + ΔW·x          W, ΔW ∈ R^(d×d)，需学 d² 个参数
```

LoRA 把 `ΔW` 分解成两个瘦矩阵的乘积，并加一个缩放系数：

```
ΔW ≈ (α/r)·B·A          B ∈ R^(d×r), A ∈ R^(r×d), r ≪ d
y = W·x + (α/r)·B·A·x   需学 2dr 个参数
```

参数量从 `d²` 降到 `2dr`。以 `d=2048, r=16` 为例：4.19M → 65.5K，约 **1/64**。

**关键认知：`r` 和 `α` 不是被学习的参数。** 它们是超参数，只决定 `A`/`B` 的**形状**，
训练中永不被梯度更新；被学习的是 `A`/`B` 里的**数值**。

> 类比 `nn.Linear(d_in, d_out)`：`d_in`/`d_out` 决定权重矩阵形状，是超参数；
> 矩阵里的数才是学出来的。`r` 的地位和 `d_out` 完全一样。
>
> 代码印证：`models/lora.py` 里 `rank` 只出现在计算 shape 的地方，
> 从不参与任何会被求导的运算。

### 1.2 「把 r 当超参，让模型自己学低秩矩阵」怎么理解

这句话拆成两半就清楚了：

| | 谁决定 | 什么时候定 |
|---|---|---|
| 秩 `r`（低秩子空间的**维度**） | 人，靠先验/搜索/显存预算 | 训练**开始前**，一旦定下整个训练期固定 |
| `A`/`B` 的**数值**（子空间的**方向**） | 梯度下降 | 训练**全程**持续更新 |

所以"模型自己学"学的是**在给定维度的子空间里往哪个方向走**，而不是学这个维度本身。

### 1.3 为什么 r 跟任务有关、却能一开始就固定

这不是 LoRA 特有的矛盾，**所有超参数都是这样**（学习率、batch size、层数都一样）：

- 超参数的最优值确实依赖任务；
- 但**在一次训练运行内**它必须固定——梯度下降没有任何机制去更新它；
- 调它的方式是**跨多次运行**做实验（人在环的超参搜索），而不是在一次运行内自适应。

**openpi 的实际做法（实证）**：全仓库只用了**两组**值，所有 LoRA 任务无差别复用，
**没有任何按任务调优**：

| 变体 | rank | alpha | α/r | 定义位置 |
|---|---|---|---|---|
| `gemma_2b_lora`（VLM 专家） | 16 | 16 | 1.0 | `gemma.py:96` |
| `gemma_300m_lora`（动作专家） | 32 | 32 | 1.0 | `gemma.py:107` |
| π0-FAST 的 VLM 专家 | 16 | 16 | 1.0 | `gemma_fast.py:68-69`（与 2b 同值） |

注意三处都是 `α = r`，所以 `scaling_value = α/r` 恒等于 **1.0** ——
这个仓库里 LoRA 的缩放系数实际从未起过放缩作用。

> 实践中常见策略（来自社区经验，非本仓库实现）：先用较大的 `r`（32/64）且令 `α=r`
> 使 `α/r=1`，对标全量微调的表达能力；再试更小的 `r` 但**保持 `α` 不变**，
> 于是 `α/r` 变大，相当于给低秩增量加权补偿。
>
> 研究方向上确有在训练中自适应重分配秩的工作（**AdaLoRA**），但 openpi 没有实现。

### 1.4 π0 里哪些层用了 LoRA

**openpi 实现。** 不是所有层都加，只加在 Gemma 的两类模块上：

| 位置 | 模块 | 代码 |
|---|---|---|
| 注意力 | `qkv_einsum` / `q_einsum` / `kv_einsum` / `out_einsum` | `gemma.py:177,185,192,238`，配 `lora_configs.get("attn")` |
| 前馈网络 | `lora.FeedForward` | `gemma.py:319-324`，配 `lora_configs.get("ffn")` |

两点值得注意：

1. **`gemma.py` 里没有独立的 `Einsum` 类**，4 处调用全部用的是 `lora.Einsum`。
   当 `lora_config=None` 时它退化成普通 einsum —— 所以"是否启用 LoRA"只是配置开关，
   不需要换模块。
2. `gemma.py` 自己定义的 `FeedForward` 类（`gemma.py:253-280`）是**死代码**，
   `Block` 永远用 `lora.FeedForward`。

**没加 LoRA 的部分**（因此在 LoRA 配置下仍是全量可训练）：
- 视觉塔 `PaliGemma/img/*`
- 机器人相关 I/O 头：`action_in_proj`、`action_out_proj`、`state_proj`、
  `time_mlp_in/out`、`action_time_mlp_in/out`

原因是 `get_freeze_filter()` 的正则是 `.*llm.*`，而 `pi0.py:91` 写的是
`self.PaliGemma = nnx.Dict(llm=llm, img=img)` —— 视觉塔挂在 `img` 下，
I/O 头是 `Pi0` 的顶层属性，都在 `.*llm.*` 的射程之外。

### 1.5 什么是 "LoRA Einsum"

**openpi 实现。** 把 LoRA 适配到 einsum 形式的通用线性变换。Gemma 的注意力投影
不是写成 `nn.Linear`，而是写成 einsum（因为要一次处理多个 head）：

```
原始:  "BTD,NDH->BTNH"     x:(B,T,D) × w:(N,D,H) → (B,T,N,H)
```

`_make_lora_eqns` 把这一个方程**动态改写成两段**，中间经过一个大小为 `rank` 的瓶颈轴 `L`：

```
第一段: "BTD,NDL->BTNL"    D 轴降到 L=rank
第二段: "BTNL,NLH->BTNH"   L 轴升回 H
```

参数形状（**实测**，`N=2, D=4, H=3, rank=2`）：

```
w        (2, 4, 3)    基础权重（冻结）
lora_a   (2, 4, 2)    降维，D → rank
lora_b   (2, 2, 3)    升维，rank → H
```

前向就是 `einsum(原方程, x, w) + (α/r) · einsum(第二段, einsum(第一段, x, lora_a), lora_b)`。
**实测**手动按公式重算与 `module.apply()` 的结果完全一致。

### 1.6 每层、每专家的 LoRA 是独立的

**通用知识 + openpi 实现。** 18 层 Transformer 在 Python 里共用同一个 `Block` 类，
但参数是独立的。机制是 `gemma.py:359-377`：

```python
self.layers = nn.scan(
    block_cls,
    variable_axes={"params": 0},      # 参数沿新增的 leading 轴堆叠
    split_rngs={"params": True},      # 每层用不同随机种子初始化
    length=self.configs[0].depth,     # depth=18
)
```

`variable_axes={"params": 0}` 的含义：所有参数多出一个 `depth=18` 的前导轴，
第 `i` 层用第 `i` 个切片。**所以 18 层各有自己独立的 `lora_a`/`lora_b`**，
两个专家（VLM / 动作）之间也各自独立。

### 1.7 冻结是三层机制，LoRA 模块本身不冻结任何东西

这是最容易误解的一点。**实测**：对一个孤立的 `lora.Einsum` 求梯度，
`w`、`lora_a`、`lora_b` **三者梯度全部非零**：

```
grad[lora_a] norm = 6.301119
grad[lora_b] norm = 4.379179
grad[w     ] norm = 6.619426    ← 基础权重也有梯度！
```

真正实现"只更新 adapter"的是**训练循环外层的过滤器**：

```python
# pi0_config.py:80-125
freeze_filter = nnx.All(
    PathRegex(".*llm.*"),              # 命中两个专家的 Gemma 权重
    nnx.Not(PathRegex(".*llm.*_1.*")), # 排除动作专家（当它自己不是 LoRA 时）
    nnx.Not(PathRegex(".*lora.*")),    # 排除 LoRA adapter 本身
)

# config.py:550-552
trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
```

然后在 `train_step` 里用 `nnx.DiffState(0, trainable_filter)` 限制求导范围——
**冻结参数连梯度都不会被计算出来**，不是算出来再丢掉。

### 1.8 参数量实测

用 `nnx.eval_shape` 构造抽象模型（只有 shape/dtype，不分配显存、不下载 checkpoint）
统计出的真实数字：

| 配置 | 参数路径数 | 冻结 | 可训练 | 可训练占比 | lora 路径 |
|---|---|---|---|---|---|
| A. 全量微调 | 50 | 0 | 3,238,048,528 | **100%** | 0 |
| B. 仅 VLM 开 LoRA | 60 | 9 (2,508,531,712) | 757,386,000 | **23.19%** | 10 |
| C. 两专家都开 LoRA | 70 | 17 (2,819,996,672) | 468,039,440 | **14.23%** | 20 |

配置 C 就是 `pi0_libero_low_mem_finetune` 的真实配置。其中 33 条
"不带 lora 但仍可训练"的路径全部是 `PaliGemma/img/*`（视觉塔），印证了 §1.4。

上游测试 `pi0_test.py` 的断言与此一致：全量微调冻结 0 个，
仅 Gemma LoRA 冻结 9 个，仅动作专家 LoRA 冻结 8 个，两者都开冻结 17 个（=9+8）。

### 1.9 LoRA 为什么省显存

**通用知识。** 一个**可训练**参数在 AdamW 下大约要占 4 份存储：

```
参数本身(fp32) + 梯度 + 一阶动量 m + 二阶动量 v
```

冻结参数只需要第 1 份，而且可以降精度。openpi 对应三处设计：

| 省在哪 | 代码位置 | 效果 |
|---|---|---|
| 冻结参数转 bfloat16 | `train.py` `init()` 里的 `state_map(..., freeze_filter, ...)` | 那 ~86% 的参数省一半 |
| 优化器状态只给可训练参数分配 | `tx.init(params.filter(trainable_filter))` | m/v 只占 14% 参数的量 |
| 限制求导范围 | `nnx.DiffState(0, trainable_filter)` | 冻结参数不产生梯度 |

### 1.10 加载非 LoRA 基座 checkpoint 时 adapter 怎么来

`pi0_base` 这类基座 checkpoint 里**没有** `lora_a`/`lora_b`。流程是：

1. `config.model.create(rng)` 先把**所有**参数随机初始化（含 LoRA adapter）；
2. `weight_loaders.CheckpointWeightLoader` 加载基座权重，对找不到的参数
   按 `missing_regex=".*lora.*"`（`weight_loaders.py:54`）留下占位；
3. `_load_weights_and_validate` 把占位的 `ShapeDtypeStruct` 剔掉；
4. `state.replace_by_pure_dict(partial_params)` 只覆盖加载到的那部分。

结果：**基座权重继承预训练值，LoRA adapter 保留随机初始化值。**

---

## 2. 超参数：学习率与 batch size

### 2.1 基本机制

**通用知识。**

```
θ ← θ - lr · ∇L(θ)
```

- **学习率**决定每一步沿梯度方向走多远；
- **batch size** 决定 `∇L` 是在多少条样本上估计的 → 决定这个估计的**噪声大小**。

| | 太小 | 太大 |
|---|---|---|
| 学习率 | 收敛慢、容易卡在平坦区 | 震荡、发散、NaN |
| batch size | 梯度噪声大（但有正则化效果） | 算力/显存成本高，泛化有时反而变差 |

**线性缩放经验法则**：batch size 放大 k 倍时，学习率大致也放大 k 倍。

### 2.2 epoch / step / batch size 的换算

**通用知识。** 设数据集 `N` 条，batch size `b`，总步数 `n`，epoch 数 `m`：

```
每个 epoch 的步数 = N / b
m = n·b / N
```

**但要注意采样顺序**：真实训练里 `shuffle=True`（`train.py` 里 `create_data_loader(..., shuffle=True)`），
**每个 epoch 会重新打乱**。所以不是"第 1 步用第 1-5 条、第 2 步用第 6-10 条……
循环回来还是 1-5"这种固定顺序；但每个 epoch 内每条样本仍然恰好被用到一次。

### 2.3 warmup + 余弦退火

**openpi 实现**（`optimizer.py` `CosineDecaySchedule`，所有 LoRA 配置也用这个默认值）：

```python
warmup_steps = 1_000
peak_lr      = 2.5e-5
decay_steps  = 30_000
decay_lr     = 2.5e-6
init_value   = peak_lr / (warmup_steps + 1)   # 从几乎 0 开始
```

学习率曲线：从 ~0 线性爬升到 `peak_lr`（1000 步），再余弦衰减到 `decay_lr`。

**为什么微调也需要 warmup**（这点容易被质疑）：不是因为参数"没有继承"，
而是因为 **Adam 的优化器状态（`m_t`/`v_t`）是全新从零开始的**。
`v_t` 还没积累到可靠的二阶统计量时，`m/√v` 这个比值可能异常大；
再叠加早期 batch 的梯度噪声，一上来就用峰值学习率容易把参数推坏。
这和"是预训练还是微调"无关。

### 2.4 三组实测对照

用真实的 `scripts/train.py` + `dummy` 微型 Gemma + 合成数据跑的 20 步对照
（日志：`notes/lora_hands_on/runs/combined.log`）：

```bash
# 只改一个变量，其余完全相同
scripts/train.py debug --exp-name=lr_baseline --batch-size=2  --lr-schedule.peak-lr=2.5e-5
scripts/train.py debug --exp-name=lr_huge     --batch-size=2  --lr-schedule.peak-lr=1.0
scripts/train.py debug --exp-name=bs_large    --batch-size=64 --lr-schedule.peak-lr=2.5e-5
# 三组都加 --num-train-steps=20 --log-interval=1 --lr-schedule.warmup-steps=0 --overwrite
```

**对照 1：学习率（batch_size=2 不变，lr 放大 4 万倍）**

| | Step 0 | Step 3 | Step 19 |
|---|---|---|---|
| lr=2.5e-5 loss | 2.6075 | 2.4000 | 2.4191 |
| lr=2.5e-5 param_norm | 477.86 | 477.86 | 477.86（20 步不动） |
| lr=1.0 loss | 2.6075 | **2682.65** | 16.74 |
| lr=1.0 param_norm | 646.90 | 1184.96 | **2441.20** |

起点完全相同（都是 2.6075，同种子同数据），但 lr=1.0 在第 3 步 loss 就炸到 2682，
`param_norm` 从 646 单调涨到 2441。后面 loss 偶尔掉下来（如 step15 的 4.80）
不是"训好了"，是震荡中偶然经过一个凑合的点。

**对照 2：batch size（lr=2.5e-5 不变，batch 放大 32 倍）**

| | grad_norm 范围 | loss 范围 |
|---|---|---|
| batch_size=2 | 5.33 ～ 16.75（跨度 ≈11.4） | 2.07 ～ 3.12，杂乱无趋势 |
| batch_size=64 | 2.23 ～ 3.10（跨度 ≈0.9） | 2.19 ～ 2.44，平滑下降 |

梯度范数的波动跨度收窄了**一个数量级**。原因很直接：batch=64 时每步梯度是 64 条
样本的平均，单条样本的噪声互相抵消；batch=2 只平均 2 条，噪声几乎原样保留。

> 这组实验数据是合成随机噪声，所以"是否真学到东西"看不出来，
> 但"梯度噪不噪"这一点的对比非常干净。

### 2.5 读日志时必须知道的两个陷阱

这两点是读 `train.py` 源码时才发现的，不看代码会把日志解读错：

**① `grad_norm` 是裁剪前的原始值。** `AdamW.create` 返回的是

```python
optax.chain(optax.clip_by_global_norm(1.0), adamw)   # optimizer.py:85
```

梯度裁剪**包在优化器内部**，而 `info["grad_norm"] = optax.global_norm(grads)`
取的是传入优化器**之前**的梯度。所以 lr=1.0 那组 `grad_norm` 能飙到 1097，
但真正用于更新的梯度已被压到全局范数 ≤ 1.0 —— 这正是
`param_norm` 呈**近似线性增长而非指数爆炸**的原因。

**② 打印的是区间平均，不是瞬时值。** `train.py` 把每步的 `info` 累积进 `infos` 列表，
到 `log_interval` 才 `stack_forest` + 求均值。默认 `log_interval=100`，
所以默认日志里的 loss 是 **100 步的平均**。要看逐步真实波动必须加 `--log-interval=1`。

### 2.6 LoRA 该不该用更高的学习率

**一个需要谨慎的论断。** 常见说法是"LoRA 的 adapter 是随机初始化的，所以可以/应该用更大的 lr"。
这个说法的常见论证方式（"全量微调的 ΔW 是继承来的"）是**错的**——
全量微调在初始时 `ΔW` 同样 ≈ 0，两边都没有"继承的 ΔW"。

更准确的区分是看**实际被优化的那个参数**从哪里出发：

| | 可训练参数 | 初始值 | 基座权重的风险 |
|---|---|---|---|
| 全量微调 | `W` 本身 | 预训练值（**有信息**） | `W` 直接暴露，大步长会破坏预训练知识 |
| LoRA | `A`/`B` | 随机噪声（**无信息**） | `W` 被冻结，架构上免疫 |

**但 openpi 并没有这么做（实证）**：`pi0_libero_low_mem_finetune` 没有覆盖
`lr_schedule`，用的是和全量微调**完全相同**的 `peak_lr=2.5e-5`。
所以"LoRA 用更高 lr"是社区经验，不是这个仓库的实现。

---

## 3. 数据归一化

### 3.1 为什么需要

**通用知识。** 真实机器人数据各维度量纲差异极大：关节角是弧度（±3.14）、
夹爪开合是 0/1、末端位置可能是米或毫米。直接喂给模型，**量纲大的维度会主导梯度，
量纲小的维度几乎学不动**。

只归一化 `state` 和 `actions` 两个字段。**图像不参与**——图像的归一化是 `/255`
这类固定规则，不依赖数据集统计。

### 3.2 两套公式，按模型类型选

**openpi 实现。** 由 `use_quantile_norm = (model_type != PI0)`（`config.py:187`）决定：

| 模型 | 公式 | 实际范围 |
|---|---|---|
| π0 | `(x - mean) / std` | 均值 0、标准差 1；约 68% 落在 [-1,1]，**可以超出** |
| π0.5 / FAST | `(x - q01) / (q99 - q01) · 2 - 1` | 1%~99% 区间**精确**映射到 [-1,1]，约 2% 离群点在外 |

所以 `norm_stats.json` 必须同时存 `mean`/`std`/`q01`/`q99` 四个量——两条路都要支持。

### 3.3 为什么不能在训练时在线算

**通用知识。** 三条理由：

1. 归一化参数必须在**整个数据集**上一致。每个 batch 各算各的，
   同一个动作在不同 batch 会被归一化成不同的数，模型学不到稳定映射；
2. 推理部署时只有单帧观测，根本没法统计，必须复用训练时的同一份
   （这也是它被打包进 checkpoint 的 `assets/` 目录的原因）；
3. 纯 I/O 密集的一次性工作，没必要每次训练重做。

**在线累积算法**（`shared/normalize.py` `RunningStats`）：

- 均值/平方均值用**按样本数加权的在线更新**，数学上与一次性全量计算**完全等价**
  → 所以 batch size 取多少都不影响结果；
- 方差 = 平方均值 - 均值²；
- 分位数用 **5000 个 bin 的直方图近似**（min/max 变大时重新分桶），所以 q01/q99 是**近似值**。

**实测验证**：对 `uniform(-1,1)` 的合成数据，算出 mean≈0、std≈0.5774、q01/q99≈±0.98 ——
`std` 的理论值是 `1/√3 = 0.5774`，完全吻合。

### 3.4 最关键的不变量：统计必须发生在流水线的正确位置

**openpi 实现。** 训练时的完整数据流水线是：

```
repack_transforms.inputs     改 key 名，把数据集字段映射成 openpi 约定
data_transforms.inputs       机器人相关处理（拼接关节、调整动作维度、增量转换…）
Normalize(norm_stats)        ★ 归一化插在这里
model_transforms.inputs      tokenize 文本、resize 图像
```

而 `compute_norm_stats.py` **只套前两段**，恰好停在 `Normalize` 之前。
这样统计出的均值/方差，正好是 `Normalize` 将来要作用的那个坐标空间里的数字。

> **少套或多套一层，归一化就会失效甚至把数据推到更离谱的范围。**
> 这是整个脚本最核心的设计约束。

### 3.5 归一化是双向的：actions 出模型还要还原

**通用知识 + openpi 实现。** `state` 只是输入，但 `actions` **既是训练时的监督目标、
又是推理时模型的输出**。模型学的是"预测归一化后的动作"，
所以推理结果必须先还原回物理单位才能发给机器人。

推理时有一对严格镜像的流水线（`policies/policy_config.py:77-89`）：

```
inputs:   repack.inputs → data_transforms.inputs → Normalize   → model_transforms.inputs
outputs:  model_transforms.outputs → Unnormalize → data_transforms.outputs → repack.outputs
```

公式互逆：`(x-mean)/std` ↔ `x·std + mean`。

**一个容易看懵的细节**：`policies/policy.py:92-95` 把明明用不到的 `state`
也塞进了模型输出字典：

```python
outputs = {"state": inputs["state"], "actions": self._sample_actions(...)}
```

原因是 `Unnormalize` 用 `strict=True`（`transforms.py:167`），
strict 模式下 `apply_tree` 会检查 **norm_stats 里的每个 key 都必须出现在数据里**
（`transforms.py:415-418`）。norm_stats 恰好有 `state` 和 `actions` 两个 key，
输出里少了 `state` 就会抛 `ValueError`。

### 3.6 归一化参数不只取决于数据集，也取决于配置

**这是个反直觉但很重要的点。** 因为统计发生在 `data_transforms` **之后**（§3.4），
量的是**变换后的张量**而非原始数据。同一份原始数据经过不同变换，统计量就不同。

**实证**：`physical-intelligence/libero` 这**同一个数据集**被 5 个配置共用，
但 `extra_delta_transform` 开关不一样：

| 配置 | `extra_delta_transform` |
|---|---|
| `pi0_libero`、`pi0_libero_low_mem_finetune`、`pi0_fast_libero`、`pi0_fast_libero_low_mem_finetune` | `True` |
| `pi05_libero` | `False` |

落盘路径的设计反过来印证了这件事：

```
assets/<config.name>/<repo_id>/norm_stats.json
       ↑ 配置名在前      ↑ 数据集在后
```

如果归一化参数只跟数据集有关，最自然的设计是 `assets/<repo_id>/` 全局共用。
但实际是**配置名在第一级**——目录结构本身就是为"同一数据集、不同统计量"准备的。

> 反过来，当多个配置**确实**共享同一归一化空间时，可以用 `asset_id`
> （`config.py:181`）显式指向同一份统计量复用。
>
> ⚠️ 坑：`compute_norm_stats.py` **写入**用 `repo_id`，训练时**读取**用 `asset_id`。
> 多数配置两者相同；但如 `AssetsConfig(asset_id="trossen")` 的配置会对不上，需手动挪文件。

---

## 4. 动作表示：绝对 vs 增量

### 4.1 DeltaActions 做什么

**openpi 实现。** `transforms.py:219` 一行：

```python
actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
```

即**把动作从"绝对目标位置"改写成"相对当前 state 的增量"**，
由一个 bool mask 控制哪些维度参与。`make_bool_mask(6, -1)` = `(T,T,T,T,T,T,F)`，
即前 6 个关节转增量、第 7 维（夹爪）保持绝对值——**夹爪永远是绝对的**。

### 4.2 对归一化统计量的影响有多大

**实测**（用真实的 `transforms.DeltaActions` + `RunningStats`，喂一段符合真实机器人
特征的平滑关节轨迹：各关节有自己的工作点偏置、相邻帧变化很小）：

```
绝对动作:  mean = [ 0.047 -0.249  0.025 -2.198  0.005  2.024  0.832]
           std  = [ 0.176  0.168  0.175  0.180  0.174  0.178  0.175]

增量动作:  mean = [ 0.002 -0.001 -0.002 -0.001  0.001  0.002  0.832]
           std  = [ 0.015  0.018  0.020  0.022  0.025  0.026  0.175]
```

- **mean**：第 4 关节从 `-2.198` → `-0.001`。绝对角度带着工作点偏置，增量天然以 0 为中心；
- **std**：前 6 维相差 **7~12 倍**（相邻帧动得少，增量方差自然小一个量级）；
- **第 7 维（夹爪，被 mask 排除）**：两边 std 都是 `0.1745`，**完全一致** ——
  正好验证只有被 mask 选中的维度变了。

### 4.3 决定因素是数据集的原始表示，不是模型

三个触发 `DeltaActions` 的开关**全部挂在 DataConfig 类上**，没有一个在模型配置上：

| 开关 | 所属类 | 默认 |
|---|---|---|
| `use_delta_joint_actions` | `LeRobotAlohaDataConfig`（`config.py:232`） | `True` |
| `extra_delta_transform` | `LeRobotLiberoDataConfig`（`config.py:289`） | `False` |
| `action_space == JOINT_POSITION` | `RLDSDroidDataConfig`（`config.py:403`） | — |

逻辑是"**原始数据是绝对的就转成增量**"。DROID 那处的注释写得很直白：
*"Data loader returns absolute joint position actions -- convert to delta actions for training."*

### 4.4 LIBERO 上 True/False 差异的真实原因

不是"相对 vs 绝对"的设计选择，而是**历史包袱 vs 正确做法**。
`config.py:332-337` 的原注释是关键：

> *In Libero, the raw actions in the dataset are already delta actions, so we \*do not\*
> need to apply a separate delta conversion.*
> *LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints
> that are trained with this extra delta transform.*

- LIBERO 原始数据**本来就是增量动作**；
- `pi05_libero` 用 `False` —— 直接用原始增量，**语义正确**；
- 四个 π0 配置用 `True` —— 在增量之上**再减一次 state**（增量的增量），
  纯粹为**兼容早期 π0 checkpoint**（`extra_` 前缀就在暗示"额外的、非必要的"）。

---

## 5. 训练工程实现

### 5.1 train.py 的三阶段结构

| 阶段 | 函数 | 职责 |
|---|---|---|
| 编排 | `main()` | 建 device mesh、checkpoint 管理器、data loader，然后进 for 循环 |
| 开局 | `init_train_state()` | 造模型、灌预训练权重、定可训练范围、分配优化器状态、算 FSDP 分片 |
| 单步 | `train_step()` | 前向算 loss → 反向算梯度 → 优化器更新，被 `jax.jit` 编成一张 XLA 图 |

### 5.2 `nnx.eval_shape` / `jax.eval_shape`：只推形状不分配显存

**通用技巧（JAX 生态）。** 构造"抽象模型"——只有 shape/dtype，不做真实计算、不分配内存。
用途：

- 秒级查看完整参数树结构，**不需要下载 checkpoint、不需要等真正的权重初始化**；
- 算 FSDP 分片方案（需要知道每个张量的形状）；
- 给 weight loader 做结构校验。

§1.8 那张参数量表就是靠这个在 CPU 上秒出的。

### 5.3 Flax NNX 的 split / merge 模式

**通用知识（Flax NNX）。** 模型被拆成两部分：

- `graphdef`（结构，静态 Python 对象）
- `state` / `params`（纯数据，可以被 jit 追踪、分片、donate）

训练时参数在 `TrainState` 里以纯数据形式流转，用的时候
`nnx.merge(state.model_def, state.params)` 合成可调用的模型。
这是让"参数"能被 JAX 变换（jit / 分片 / 求导过滤）处理的前提。

### 5.4 可复现性：`jax.random.fold_in`

**通用技巧。**

```python
train_rng = jax.random.fold_in(rng, state.step)
```

用 step 数"折"出本步的种子。既保证每步随机性不同（流匹配的噪声采样、dropout），
又保证**完全可复现**——同样的 seed + 同样的 step 必然得到同样的随机数。

**实测印证**：隔了很久重跑 `debug` 配置，`Step 0` 依然是
`grad_norm=10.1090, loss=2.6075`，与之前那轮完全一致。

### 5.5 显存优化手段

| 手段 | 代码 | 原理 |
|---|---|---|
| 冻结参数降精度 | `state_map(params, freeze_filter, → bfloat16)` | 不更新的参数不需要 fp32 |
| 优化器状态只给可训练参数 | `tx.init(params.filter(trainable_filter))` | AdamW 的 m/v 是 2× 参数量 |
| 限制求导范围 | `nnx.DiffState(0, trainable_filter)` | 冻结参数不产生梯度 |
| 缓冲区捐赠 | `jax.jit(..., donate_argnums=(1,))` | 新参数直接写在旧参数内存上，省一份完整副本 |
| FSDP 分片 | `sharding.fsdp_sharding(...)` | 参数本身切开分散到多卡 |

### 5.6 FSDP 分片策略

**openpi 实现**（`sharding.py`）。设备网格是二维的：

```
mesh_shape = (device_count // fsdp_devices, fsdp_devices)
轴名         = (BATCH_AXIS,                  FSDP_AXIS)
DATA_AXIS   = (BATCH_AXIS, FSDP_AXIS)        # 数据沿两个轴一起切
```

- `fsdp_devices=1`（默认）→ 纯数据并行；
- 调大 → **参数本身**也切开，用于单卡放不下的全量微调（π0.5 全量微调需 >70GB）。

`fsdp_sharding` 的具体策略：标量/向量（ndim<2）复制、小于 4MiB 的张量复制、
其余沿**最大的且能被 fsdp 维整除的轴**切分，找不到合适轴就复制。

### 5.7 EMA（指数滑动平均）

**通用知识。**

```python
ema_params = ema_decay · ema_params + (1 - ema_decay) · new_params
```

维护一份"慢速跟随"的参数副本，通常比瞬时参数更平滑、推理效果更好。
**代价是额外一份完整参数的显存**。

openpi 默认 `ema_decay=0.99`，但 `pi0_libero_low_mem_finetune` 显式设成
`None`（`config.py:697`）来省显存；`pi05_libero` 用 `0.999`。

存 checkpoint 时如果有 EMA，**导出给推理用的就是 EMA 版本**
（`checkpoints.py:145-152` 的 `_split_params`）。

### 5.8 param_norm 这个指标在量什么

只统计"权重矩阵"，刻意排除 `bias`/`scale`/各种 embedding，并要求 `ndim > 1`：

```python
nnx.All(
    nnx.Param,
    nnx.Not(PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
    lambda _, x: x.value.ndim > 1,
)
```

这样它才是干净的"模型权重整体幅度"指标：**突然暴涨说明训练发散**；
**长期一动不动说明更新量相对权重本身小到可以忽略**（§2.4 基线那组 20 步
停在 477.8596 就是后者）。

### 5.9 异步数据预取

```python
with sharding.set_mesh(mesh):
    train_state, info = ptrain_step(train_rng, train_state, batch)
...
batch = next(data_iter)   # ← 放在训练之后
```

JAX 是**异步派发**的：`ptrain_step` 返回时 GPU 上的计算可能还没做完。
把取下一个 batch 放在后面，读盘/解码就能在主机侧与 GPU 计算并行进行，
相当于免费的数据预取。

### 5.10 checkpoint 机制要点

- **目录冲突处理**：`--overwrite` 删掉重建 / `--resume` 续训 / 都不给**直接报错**
  （`checkpoints.py:25-36`）；两者不能同时为真（`config.py` `__post_init__` 校验）。
- **保留策略**：默认只留最新 1 个，但 `step % keep_period == 0` 的会被永久保留
  （`keep_period` 默认 5000）。
- **异步写盘**：主循环结束必须 `wait_until_finished()`，否则进程退出会导致
  最后一个 checkpoint 损坏或丢失。
- **norm_stats 会被一起打包**进 checkpoint 的 `assets/<asset_id>/`
  （`checkpoints.py:71-76`），这样推理端拿到 checkpoint 就自带归一化参数。

---

## 6. 几个被推翻的直觉

这一节单独列出来，因为这些是本次学习过程中**真的被代码推翻**的说法。

### 6.1 ❌「LoRA 模块自己会冻结基础权重」

**错。** 实测对孤立的 `lora.Einsum` 求梯度，`w` 的梯度范数是 6.619426，非零。
冻结是外层 `freeze_filter` / `trainable_filter` 做的事（§1.7）。

### 6.2 ❌「LoRA 可以用更高的学习率，因为它的 ΔW 不是继承来的」

**论证方式错。** 全量微调初始时 `ΔW` 同样 ≈ 0，两边都没有"继承的 ΔW"。
正确的区分是看实际被优化的参数从**有信息的点**（全量微调的 `W`）
还是**无信息的随机点**（LoRA 的 `A`/`B`）出发（§2.6）。

而且**openpi 实际并没有给 LoRA 用更高的 lr** —— 实测
`pi0_libero_low_mem_finetune` 用的是和全量微调完全相同的 `peak_lr=2.5e-5`。

### 6.3 ❌「归一化参数只跟数据集有关」

**错。** 统计发生在 `data_transforms` 之后，量的是变换后的张量。
同一份 LIBERO 数据，`pi0_libero` 和 `pi05_libero` 的 `norm_stats.json`
因为一个 `extra_delta_transform` 开关而相差近一个数量级（§3.6、§4.2）。
落盘路径把 `config.name` 放在第一级就是为此准备的。

### 6.4 ❌「π0 用相对动作，π0.5 用绝对动作」

**错，两个方向都有反例：**

- `pi05_full_droid_finetune` 是 π0.5（`pi05=True`），却设了
  `action_space=JOINT_POSITION`，**触发了** `DeltaActions` → π0.5 在 DROID 上用增量动作；
- `pi0_aloha_sim` 是 π0，却设了 `use_delta_joint_actions=False` → π0 也有用绝对动作的配置。

真正的决定因素是**原始数据集用什么表示**，与模型无关（§4.3）。

### 6.5 ❌「日志里的 grad_norm 就是实际用于更新的梯度范数」

**错。** 梯度裁剪 `clip_by_global_norm(1.0)` 包在优化器内部，
而 `info["grad_norm"]` 取的是传入优化器之前的值（§2.5）。

### 6.6 ❌「日志里的 loss 是当前这一步的 loss」

**默认情况下错。** 是上一个 `log_interval` 区间内所有步的**平均值**，
默认 100 步平均（§2.5）。

### 6.7 ❌「配置名都可以当位置参数传」

**错，三个脚本三种风格**（实测 usage 输出）：

| 脚本 | 配置名怎么传 | 底层 |
|---|---|---|
| `train.py` | **位置参数** `train.py pi0_libero...` | `tyro.extras.overridable_config_cli`，每个配置是一个子命令 |
| `compute_norm_stats.py` | **`--config-name=xxx`** | `tyro.cli(main)`，从函数签名推导 |
| `serve_policy.py` | **`--policy.config=xxx`** | `tyro.cli(Args)`，`policy:checkpoint` 只是联合类型的子命令选择器 |

---

## 7. 配套可跑脚本

都在 `notes/lora_hands_on/`，用的都是**真实的 openpi 生产代码**（非重新实现），
只是把规模缩小到 CPU 秒级可跑。

| 文件 | 验证什么 | 跑法 |
|---|---|---|
| `01_lora_math_demo.py` | LoRA 前向公式、`_make_lora_eqns` 的方程改写、`w`/`lora_a`/`lora_b` 的梯度 | `uv run python notes/lora_hands_on/01_lora_math_demo.py` |
| `02_real_model_param_tree.py` | 真实 Pi0 上 `freeze_filter` 筛出了哪些参数、§1.8 那张参数量表 | `JAX_PLATFORMS=cpu uv run python notes/lora_hands_on/02_real_model_param_tree.py` |
| `runs/combined.log` | §2.4 三组 lr / batch size 对照实验的原始日志 | 见 §2.4 的命令 |

另外两个脚本被加了逐行中文注释，可以当作带导读的源码阅读：

- `scripts/compute_norm_stats.py` —— 归一化统计量的计算
- `scripts/train.py` —— 训练主循环

### 快速自建实验的技巧

想自己验证训练行为而不想等真实数据集下载 / 大模型加载，用 `debug` 配置：

```python
TrainConfig(
    name="debug",
    data=FakeDataConfig(),                                    # repo_id="fake"，合成随机数据
    model=Pi0Config(paligemma_variant="dummy",                # width=64, depth=4,
                    action_expert_variant="dummy"),           # mlp_dim=128, heads=8, head_dim=16
    batch_size=2, num_train_steps=10, wandb_enabled=False,
)
```

配合 `JAX_PLATFORMS=cpu` 和 tyro 的嵌套覆盖语法，几十秒就能跑完一组对照：

```bash
JAX_PLATFORMS=cpu uv run scripts/train.py debug \
    --exp-name=my_test --overwrite --num-train-steps=20 --log-interval=1 \
    --batch-size=64 --lr-schedule.warmup-steps=0 --lr-schedule.peak-lr=2.5e-5
```

### 环境准备

`uv` 会自动建虚拟环境，**不需要手动 `python -m venv` 或 `uv venv`**：
只要目录里有 `pyproject.toml`，`uv sync` / `uv run` 发现没有 `.venv`
就会按 `requires-python` 自动创建并装好 `uv.lock` 锁定的依赖。
后续所有 `uv run xxx` 都在这个环境里执行，也不需要手动 `source .venv/bin/activate`。
