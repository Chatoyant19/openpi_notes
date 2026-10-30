# openpi 代码库深度解析：功能全景、训练流程，以及 π0 / π0-FAST / π0.5 架构对比

> 内部参考文档。建议对照源码树 `/local/mnt2/workspace/danwa/project/VLA/openpi` 阅读。
> 所有引用均为相对仓库根目录的 `file:line`。本文档对应 `main` 分支、commit `215abfb` 时的代码状态。

---

## 目录

- [0. 这个仓库是什么](#0-这个仓库是什么)
- [1. 仓库结构导览](#1-仓库结构导览)
- [2. 模型架构 —— 核心对比](#2-模型架构--核心对比)
  - [2.1 共享 backbone：三个模型完全一致](#21-共享-backbone三个模型完全一致)
  - [2.2 π0 vs π0.5 —— 字面意义上的同一个类，靠一个 flag 区分](#22-π0-vs-π05--字面意义上的同一个类靠一个-flag-区分)
  - [2.3 π0-FAST —— 真正独立的另一个模型类](#23-π0-fast--真正独立的另一个模型类)
  - [2.4 Tokenization —— 另一个主要分叉点](#24-tokenization--另一个主要分叉点)
  - [2.5 LoRA 支持情况](#25-lora-支持情况)
  - [2.6 配置项默认值](#26-配置项默认值)
  - [2.7 PyTorch 复现版（`models_pytorch/`）](#27-pytorch-复现版modelspytorch)
  - [2.8 值得一读的设计意图注释（原文引用）](#28-值得一读的设计意图注释原文引用)
  - [2.9 一页速查对比表](#29-一页速查对比表)
- [3. 训练流程](#3-训练流程)
  - [3.1 配置系统架构](#31-配置系统架构)
  - [3.2 已注册的训练配置（按模型 / 平台归类）](#32-已注册的训练配置按模型--平台归类)
  - [3.3 JAX 训练循环（`scripts/train.py`，280 行）](#33-jax-训练循环scriptstrainpy280-行)
  - [3.4 PyTorch 训练循环（`scripts/train_pytorch.py`，632 行）—— 与 JAX 版已核实的差距](#34-pytorch-训练循环scriptstrainpytorchpy632-行-与-jax-版已核实的差距)
  - [3.5 LoRA 接线细节](#35-lora-接线细节)
  - [3.6 数据加载流水线](#36-数据加载流水线)
  - [3.7 Checkpoint 格式与 EMA（JAX）](#37-checkpoint-格式与-emajax)
- [4. 数据 / Policy 层](#4-数据--policy-层)
  - [4.1 `Policy.infer()` 全链路（`src/openpi/policies/policy.py`）](#41-policyinfer-全链路srcopenpipoliciespolicypy)
  - [4.2 `create_trained_policy` 的接线（`src/openpi/policies/policy_config.py:16-94`）](#42-createtrainedpolicy-的接线srcopenpipoliciespolicyconfigpy16-94)
  - [4.3 各平台变换对比](#43-各平台变换对比)
  - [4.4 归一化 —— 分位数 vs z-score](#44-归一化--分位数-vs-z-score)
  - [4.5 Checkpoint 下载与缓存（`src/openpi/shared/download.py`）](#45-checkpoint-下载与缓存srcopenpishareddownloadpy)
  - [4.6 `nnx_utils.py`](#46-nnxutilspy)
- [5. 推理服务与远程推理基础设施](#5-推理服务与远程推理基础设施)
  - [5.1 整体架构](#51-整体架构)
  - [5.2 `scripts/serve_policy.py` —— CLI 与启动流程](#52-scriptsservepolicypy--cli-与启动流程)
  - [5.3 `action_chunk_broker.py` —— 动作块分发的确切逻辑](#53-actionchunkbrokerpy--动作块分发的确切逻辑)
  - [5.4 序列化：msgpack + numpy](#54-序列化msgpack--numpy)
  - [5.5 Docker 环境（`docs/docker.md`）](#55-docker-环境docsdockermd)
  - [5.6 Norm stats 复用机制（`docs/norm_stats.md`）](#56-norm-stats-复用机制docsnormstatsmd)
  - [5.7 `examples/simple_client` —— 无机器人也能测](#57-examplessimpleclient--无机器人也能测)
- [6. 示例与第三方集成](#6-示例与第三方集成)
- [7. 文件速查索引](#7-文件速查索引)
- [8. 功能清单 —— 这个仓库到底能做什么](#8-功能清单--这个仓库到底能做什么)
  - [8.1 模型与 Checkpoint](#81-模型与-checkpoint)
  - [8.2 训练与微调能力](#82-训练与微调能力)
  - [8.3 数据处理与格式转换](#83-数据处理与格式转换)
  - [8.4 推理与部署](#84-推理与部署)
  - [8.5 多机器人平台集成（示例）](#85-多机器人平台集成示例)
  - [8.6 Notebook / 可视化调试工具](#86-notebook--可视化调试工具)
  - [8.7 工程化 / 开发者工具](#87-工程化--开发者工具)
- [9. 实操指南：没有实机，从微调到部署跑一遍完整链路](#9-实操指南没有实机从微调到部署跑一遍完整链路)
  - [9.1 这台机器的现状核对](#91-这台机器的现状核对)
  - [9.2 两条路径共用的环境准备](#92-两条路径共用的环境准备)
  - [9.3 路径 A：快速验证全链路（几分钟，几十步）](#93-路径-a快速验证全链路几分钟几十步)
  - [9.4 路径 B：真实 LoRA 微调 + LIBERO 仿真闭环部署](#94-路径-b真实-lora-微调--libero-仿真闭环部署)
  - [9.5 两条路径的关键差异与常见坑](#95-两条路径的关键差异与常见坑)

---

## 0. 这个仓库是什么

`openpi`（Physical Intelligence 出品）开源了三代 **视觉-语言-动作（Vision-Language-Action, VLA）** 机器人策略模型，
并附带了使用它们所需的完整训练、微调与推理服务基础设施：

| 模型 | 论文/博客 | 动作头（Action Head） | 本仓库中的支持情况 |
|---|---|---|---|
| **π0** | [pi0](https://www.physicalintelligence.company/blog/pi0) | Flow matching（连续的、类扩散式去噪） | JAX + PyTorch 完整支持 |
| **π0-FAST** | [FAST](https://www.physicalintelligence.company/research/fast) | 自回归 next-token 预测，预测的是 FAST 压缩后的动作 token | 仅 JAX（无 PyTorch 实现） |
| **π0.5** | [pi05](https://www.physicalintelligence.company/blog/pi05) | Flow matching，与 π0 是同一个头；训练时使用了 [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation)（属于数据/训练流程层面的技术，**本仓库的模型代码中并未实现**，见 §2.2c） | JAX + PyTorch 完整支持（仅 flow matching 头） |

仓库提供了 base checkpoint（在 10k+ 小时机器人数据上预训练）和针对具体机器人微调的 "expert" checkpoint
（ALOHA、DROID、LIBERO），以及每个平台的示例集成代码。

---

## 1. 仓库结构导览

```
openpi/
├── src/openpi/
│   ├── models/            # JAX 模型定义：pi0、pi0-FAST、共享的 VLM backbone
│   ├── models_pytorch/    # PyTorch 复现版（仅 π0 / π0.5）
│   ├── training/          # 配置注册表、data loader、checkpoint、optimizer
│   ├── policies/          # 各机器人平台的输入/输出变换、Policy 封装
│   ├── serving/           # WebSocket policy server
│   └── shared/            # normalize.py、image_tools.py、download.py、nnx_utils.py
├── scripts/
│   ├── train.py             # JAX 训练入口
│   ├── train_pytorch.py     # PyTorch 训练入口
│   ├── serve_policy.py      # Policy server 命令行入口
│   └── compute_norm_stats.py
├── packages/openpi-client/  # 轻量级机器人端客户端（仅依赖 websockets+msgpack+numpy）
├── examples/                # aloha_real、aloha_sim、droid、libero、ur5、simple_client
├── third_party/             # aloha（ROS 驱动 fork）、libero（仿真 benchmark）两个 submodule
└── docs/                    # docker.md、norm_stats.md、remote_inference.md（+ 本文档）
```

---

## 2. 模型架构 —— 核心对比

### 2.1 共享 backbone：三个模型完全一致

三个模型使用的 **视觉-语言 backbone 是完全相同的**：一个从零搭建的 PaliGemma 风格模型
= **SigLIP `So400m/14` 视觉编码器** + **Gemma-2B 语言塔**。

- 视觉编码器：`src/openpi/models/siglip.py:293-373`，variant 为 `"So400m/14"`
  → width 1152、depth 27、mlp_dim 4304、num_heads 16、patch 14×14、`pool_type="none"`
  （返回逐 patch 的 token，不做 pooling —— 直接作为 VLM 的 image token 使用）。
  在 `pi0.py:81-90` 与 `pi0_fast.py:147-156` 中以完全相同的方式实例化。
- `src/openpi/models/vit.py` 虽然存在，但是 **死代码** —— 没有任何模型文件 import 它。
- 语言塔的各个 variant 定义在 `src/openpi/models/gemma.py:58-109`：

  | variant | width | depth | mlp_dim | num_heads | num_kv_heads | head_dim | 备注 |
  |---|---|---|---|---|---|---|---|
  | `gemma_300m` | 1024 | 18 | 4096 | 8 | 1 | 256 | 注释写明 "311M params"（`gemma.py:69-70`），在 π0/π0.5 中作为 **action expert** |
  | `gemma_2b` | 2048 | 18 | 16384 | 8 | 1 | 256 | PaliGemma LLM backbone，三个模型都用它 |
  | `*_lora` 系列 | 维度同上 | | | | | | 额外挂 LoRA adapter，rank 16/α16（2b）或 rank 32/α32（300m） |

  仓库里没有任何地方明确写出总参数量，但按 config 推算：
  Gemma-2B（~2B）+ SigLIP-So400m（~400M）+ action expert 300M（~311M）≈ 3B 出头，
  与 PI 公开的 ~3.3B 数字吻合 —— 注意这是**从 config 推断出来的，不是仓库里写明的数字**。

### 2.2 π0 vs π0.5 —— 字面意义上的同一个类，靠一个 flag 区分

**关键结论：代码里根本不存在 `Pi05` 这个类。** 两者都是 `Pi0`
（`src/openpi/models/pi0.py:66`），通过 `Pi0Config.pi05: bool = False`
（`src/openpi/models/pi0_config.py:31`）来切换。`Pi0Config.model_type` 也仅仅根据这个 flag
返回 `ModelType.PI0` 或 `ModelType.PI05`（`pi0_config.py:50-55`）。

这个 flag 在 `Pi0.__init__` / `embed_suffix` 内部实际上只控制两件结构性的事情：

**(a) 机器人状态（state）如何进入模型**

| | π0（`pi05=False`） | π0.5（`pi05=True`） |
|---|---|---|
| 状态表示 | 连续的 suffix token，经由 `self.state_proj`（`Linear(action_dim, width)`） | **离散化后拼进语言 prompt** 里，作为文本 token（完全没有连续 state token） |
| 代码路径 | `embed_suffix` 中 `if not self.pi05:` 分支会追加一个 `state_token`（`pi0.py:151-157`） | 该分支被整段跳过 |
| Tokenizer | `PaligemmaTokenizer`，走 `state=None` 分支（`tokenizer.py:30-33`，注释："the Pi0 format, where the state is part of the continuous action expert input"） | `PaligemmaTokenizer`，走 `state is not None` 分支（`tokenizer.py:24-29`）：用 `np.digitize` 把 state 离散成 256 个 bin，拼出 prompt `"Task: {text}, State: {state_str};\nAction: "` —— 注释："the Pi05 format, where the state is part of the discrete language input" |
| 谁在消费这个开关 | `discrete_state_input` 配置项（`pi0_config.py:33,40-41`），默认跟随 `pi05` 的值，但可以独立覆盖；实际消费点在 `ModelTransformFactory`（`src/openpi/training/config.py:126-138`），**模型类本身并不直接读它** |
| `max_token_len` 默认值 | 48 | 200（因为 prompt 里现在还要塞离散化的 state） |

一个真实配置里的细节值得注意：`pi05_libero`（`training/config.py:745`）设的是
`Pi0Config(pi05=True, discrete_state_input=False)` —— 这是一个 **完全没有 state 输入** 的 π0.5
（既没有连续 suffix token，因为 `pi05=True` 跳过了那条分支；也没有离散 token，因为被显式关掉了）。
对比 `pi05_droid`（`config.py:630-631`）没有覆盖 `discrete_state_input`，保持默认 `True`，
所以 DROID 版 π0.5 是有离散 state token 的。

**(b) Flow matching 的时间步（timestep）如何注入**

| | π0 | π0.5 |
|---|---|---|
| 机制 | 先 concat 再过 MLP：正弦-余弦时间嵌入沿 action horizon 广播，与 action token 嵌入拼接，然后过 `action_time_mlp_in → swish → action_time_mlp_out`（`pi0.py:170-177`） | **AdaRMSNorm**：时间嵌入先走自己的 `time_mlp_in → swish → time_mlp_out → swish`（`pi0.py:163-167`），然后作为 `adarms_cond` **走带外通道** 送进 action expert 的归一化层 |
| 结果 | `adarms_cond = None` | `adarms_cond = time_emb` |
| 归一化层 | 标准 RMSNorm | `RMSNorm.__call__(self, x, cond)`（`gemma.py:112-131`）：当 `cond is not None` 时，计算 `modulation = Dense(3*width)(cond)`，拆成 `scale, shift, gate`；输出为 `normed*(1+scale)+shift`；`gate` 随后用于 **门控残差** `x + y*gate`，取代普通的残差相加（`_gated_residual`, `gemma.py:453-459`） |
| 作用范围 | 不适用 | AdaRMS **只作用在 action expert 这一路 mixture 分支上**，即使在 π0.5 中也绝不作用于 PaliGemma/VLM 分支 —— 通过 `llm.lazy_init(..., use_adarms=[False, True] if config.pi05 else [False, False])`（`pi0.py:73-80`）连线 |

**(c) "Knowledge insulation" 在模型代码里看不到。** 对整个 model 目录 grep
`stop_gradient`/`detach`，没有任何与 π0.5 训练相关的命中（唯一的命中在一个无关的 FSQ baseline
tokenizer 里，`models/utils/fsq_tokenizer.py:113,218,223`）。README 明确写道
（`README.md:8`）：*"Note that, in this repository, we currently only support the flow
matching head for both π0.5 training and inference."*
Knowledge insulation —— 在 PI 的论文中是指训练一个离散/FAST 头、并阻断其梯度流回 VLM backbone，
同时混合另一路会正常回传梯度的 flow matching 头 —— 属于 **原论文的训练方法论，openpi 的模型代码里没有实现**。
你读 `pi0.py` 是找不到它的；它纯粹是数据/训练流程层面的技术，本次开源并未复现。

### 2.3 π0-FAST —— 真正独立的另一个模型类

`Pi0FAST`（`src/openpi/models/pi0_fast.py:134`）是一个**独立的** `BaseModel` 子类
—— 一个 decoder-only、GPT 风格的自回归 next-token 预测器，词表就是 PaliGemma 的词表。
它在架构上就是一个语言模型，而不是扩散/flow 头。

- **没有 action expert 权重。** 只有一个 LLM：`self.PaliGemma.llm`，即单 expert 的
  `gemma_fast.Module`（`pi0_fast.py:139-146`），外加 SigLIP 视觉塔。`Pi0FAST.__init__`
  里不存在 `action_in_proj`/`action_out_proj`/第二个 Gemma config。动作就是词表里的 token。
- **`embed_inputs`**（`pi0_fast.py:159-195`）：把 image token + tokenized prompt
  （训练时其中已经由 tokenizer 追加了 FAST 编码的动作 token）拼成一条序列，
  并按逐 token 的 `token_ar_mask` 构造自回归 attention mask。
- **`compute_loss`**（`pi0_fast.py:197-233`）：标准的错位目标交叉熵 LM loss，
  通过 `token_loss_mask` 限制只在动作 token 区域计算。对比 π0/π0.5 的 flow matching MSE loss
  （`pi0.py:214`：`jnp.mean(jnp.square(v_t - u_t))`）。
- **`sample_actions`**（`pi0_fast.py:235-313`）：真正的增量自回归解码 ——
  先用 prompt 预填 KV cache，然后 `jax.lax.while_loop` 循环，每次采样一个 token
  （贪心 argmax 或带 temperature 的 categorical），遇到 `PALIGEMMA_EOS_TOKEN = 1`
  （`pi0_fast.py:20`）或达到 `max_decoding_steps=256` 才停。这要求 `gemma_fast.py`
  提供真正的单 token 追加式 KV cache（`_init_cache`/`_update_cache`,
  `gemma_fast.py:165-183`）—— 而 `gemma.py` 的 mixture-of-experts attention 不需要这个，
  因为 π0/π0.5 是在 `num_steps` 个 Euler 积分步里**并行地**对整个 action horizon 长度的 suffix
  做去噪（`pi0.py:239-271`），从不逐 token 解码。
- **默认 image key 不同**：`base_0_rgb, base_1_rgb, wrist_0_rgb`
  （`pi0_fast.py:107-116`），而 π0/π0.5 是 `base_0_rgb, left_wrist_0_rgb,
  right_wrist_0_rgb`（`pi0_config.py:70-79`）。

**为什么有两个 Gemma 文件（`gemma.py` vs `gemma_fast.py`）？**

| | `gemma.py`（π0 / π0.5） | `gemma_fast.py`（π0-FAST） |
|---|---|---|
| Expert | **Mixture of experts** —— `Attention`/`Block`/`Module` 接收 `configs: Sequence[Config]` 和按 expert 分组的 token 数组列表；各 expert 共享 self-attention 的 QK/V 计算，但权重矩阵各自独立（`gemma.py:158-249, 283-333, 339-421`） | 只有单 expert —— 经典 decoder-only Gemma（`gemma_fast.py:124-224, 227-272, 278-418`） |
| AdaRMS | 有（`gemma.py:112-131`） | 无 —— 它的 `RMSNorm` 不接收 `cond` 参数（`gemma_fast.py:87-98`） |
| KV cache | 静态 prefix cache，只拼接一次（从不增量追加 —— flow matching 不做逐 token 解码） | 真正的增量 cache，用 `jax.lax.dynamic_update_slice` 单 token 追加（`gemma_fast.py:165-183`）—— `sample_actions` 的解码循环需要它 |
| LoRA | 有，通过 `lora.Einsum`/`lora.FeedForward`（`gemma.py:37`） | 有，同一个 `lora.py` 模块（`gemma_fast.py:29`） |
| 文件 docstring | — | `gemma_fast.py:15-18`：*"Gemma model implementation from big_vision/models/ppp/gemma.py... Used for FAST autoregressive policies."* |

关于 checkpoint 加载兼容性的命名约定，注释解释得很清楚（`gemma.py:444-450`）：
*"we want the first expert's weights to have no suffix... so that they can be loaded
seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a
suffix... and their weights will be initialized from scratch. in practice, we only use
two experts -- PaliGemma, and the action expert."*
这正是 `get_freeze_filter`（`pi0_config.py:88-117`，正则 `.*llm.*_1.*`）
能够精确圈出 action expert 专属参数的依据。

### 2.4 Tokenization —— 另一个主要分叉点

`src/openpi/models/tokenizer.py`：

- **`PaligemmaTokenizer`**（`tokenizer.py:14-48`）—— π0 **和** π0.5 共用
  （在 `ModelTransformFactory` 里选择，`training/config.py:113-138`）。
  按是否传入 `state` 分叉：传了 → π0.5 的离散 state 格式；没传 → π0 的连续 state 格式（见 §2.2a）。
- **`FASTTokenizer`**（`tokenizer.py:51-139`）—— 仅 π0-FAST 使用。总是构造
  `"Task: {text}, State: {state_str};\n"`（**无条件**离散化 state，`tokenizer.py:70-74`），
  训练时再追加 `"Action: " + <FAST 压缩后的 token> + "|"`。它调用外部 HF
  `physical-intelligence/fast` 的 `AutoProcessor`（一个类 BPE 的离散动作压缩器）
  把 action chunk 压成很短的 token 序列，然后把这些 FAST 词表 id 重映射到
  PaliGemma 257k 词表末尾预留的 128 个槽位里（`_act_tokens_to_paligemma_tokens`,
  `tokenizer.py:136-139`）。它还会产出 `ar_mask`（prefix 上为 0 = 双向，
  动作 postfix 上为 1 = 因果）和 `loss_mask`（只在动作 postfix 上算 loss）——
  这两者填充到 `Observation.token_ar_mask`/`token_loss_mask`（`model.py:104-107`），
  而这两个字段**只有** π0-FAST 会用到。
- `BinningTokenizer`/`FSQTokenizer`（`tokenizer.py:148-372`）被明确标注为
  仅供 RoboArena baseline 使用，任何 pi0 系模型都不用。

### 2.5 LoRA 支持情况

| 模型 | 是否支持 | 机制 |
|---|---|---|
| π0 | 支持，PaliGemma 分支与 action expert 分支可独立开启 | `Pi0Config.get_freeze_filter()`（`pi0_config.py:88-117`） |
| π0-FAST | 支持，仅 PaliGemma 分支（没有 action expert 可加 LoRA） | `Pi0FASTConfig.get_freeze_filter()`（`pi0_fast.py:127-131`） |
| π0.5 | 注册表里不存在任何 `pi05_*_lora` 配置（机制本身存在，但 π0.5 没用） | — |
| PyTorch（任意模型） | **不支持** —— 显式不支持（`README.md:197`） | — |

LoRA 并不是一个独立的"模式"，而是在 Gemma variant 层面实现的
（`"gemma_2b_lora"`/`"gemma_300m_lora"` 这类字符串会挂上 `LoRAConfig`，
`gemma.py:58-107`），再配一个 freeze filter 冻结除 `.*lora.*` 以外的一切。
`TrainConfig.trainable_filter`（`training/config.py:549-552`）
= `nnx.All(nnx.Param, nnx.Not(freeze_filter))`，被训练步用来限制梯度/优化器更新范围。
`weight_loaders.CheckpointWeightLoader`（`weight_loaders.py:50-54`）
会把 checkpoint 中缺失的 `.*lora.*` 参数从全新初始化的参数里补进来，
因为 base checkpoint 本身不含 LoRA 权重。

### 2.6 配置项默认值

**`Pi0Config`**（π0/π0.5 共用的结构体，`pi0_config.py:18-35`）：

| 字段 | 默认值 |
|---|---|
| `dtype` | `"bfloat16"` |
| `paligemma_variant` | `"gemma_2b"` |
| `action_expert_variant` | `"gemma_300m"` |
| `action_dim` | 32 |
| `action_horizon` | 50 |
| `max_token_len` | `None` → 在 `__post_init__` 里解析为 48（π0）/ 200（π0.5） |
| `pi05` | `False` |
| `discrete_state_input` | `None` → 默认取 `pi05` 的值 |
| `pytorch_compile_mode` | `"max-autotune"` |

**`Pi0FASTConfig`**（`pi0_fast.py:76-89`）：

| 字段 | 默认值 |
|---|---|
| `dtype` | `"bfloat16"` |
| `paligemma_variant` | `"gemma_2b"`（没有独立的 action expert variant 字段 —— 因为不存在 action expert） |
| `action_dim` | 32 |
| `action_horizon` | 32 |
| `max_token_len` | 250 |

### 2.7 PyTorch 复现版（`models_pytorch/`）

- `PI0Pytorch`（`models_pytorch/pi0_pytorch.py:84-172`）**同时支持 π0 和 π0.5**，
  用的是与 JAX 版一模一样的 `self.pi05 = config.pi05` 分支模式
  （`pi0_pytorch.py:88-109`，`embed_suffix` 在 238-315 行）。
  **π0-FAST 没有任何 PyTorch 实现**（已确认：不存在 `pi0_fast_pytorch.py`，
  且 `README.md:194` 把它列为不支持）。
- 双流 Gemma + action expert 的 transformer 是通过包装 HuggingFace `transformers`
  的类（`PaliGemmaForConditionalGeneration`、`GemmaForCausalLM`）实现的，
  封装在 `PaliGemmaWithExpertModel`（`models_pytorch/gemma_pytorch.py:11-280`）里，
  而非从零手写。这就要求 **给已安装的 `transformers` 包打补丁**
  （`README.md:205-212`，补丁文件在 `src/openpi/models_pytorch/transformers_replace/`），
  以加入三项能力：(1) AdaRMS 支持，(2) 激活值精度控制，(3) 允许使用 KV cache 而不更新它。
  **警告**（`README.md:210-212`）：在 uv 默认的 hardlink 模式下，这会永久性地
  改动 uv cache 里共享的 `transformers` 包；需要用 `uv cache clean transformers` 撤销。
- AdaRMS 通过 `use_adarms`/`adarms_cond_dim` 字段，配合手写的逐层 forward
  （`compute_layer_complete`, `gemma_pytorch.py:157-237`）接进去 ——
  该函数手动调用 `input_layernorm(hidden_states, cond=adarms_cond[i])`
  并施加与 JAX 版相同的门控残差，绕开了 HF 自带的 `GemmaModel.forward`。
- 精度处理：`to_bfloat16_for_selected_params`（`gemma_pytorch.py:62-82`）
  先把整个模型转成 bf16，再把一个固定清单里的参数（视觉 patch/position embedding、
  所有 `input_layernorm`/`post_attention_layernorm`、最后的 `model.norm`）
  选择性地上转回 float32 以保证数值稳定 —— 相当于把 JAX 那套"大部分 bf16、少数 fp32"
  的规则显式硬编码了一遍。PyTorch 的训练精度是 **全有或全无**
  （整体 bf16 或整体 fp32，由 `pytorch_training_precision` 配置项控制）；
  JAX 的默认则是真正的混合精度（权重/梯度 fp32，激活 bf16）。
- `sample_actions` 被 `torch.compile(mode=config.pytorch_compile_mode)` 包装
  （`pi0_pytorch.py:112-113`）。
- **PyTorch 端已在代码层面确认不支持的能力**（不只是文档声明）：
  FSDP（`train_pytorch.py` 全文零引用）、LoRA（freeze filter/`trainable_filter`
  从未被读取；`gemma_pytorch.py`/`pi0_pytorch.py` 里连 `"lora"` 字符串都没有）、
  EMA（`train_pytorch.py:499` 直接打日志 `"EMA is not supported for PyTorch training"`）、
  真正的混合精度，以及 DROID RLDS 数据加载（`data_loader.py:367-368` 抛
  `NotImplementedError`）。

### 2.8 值得一读的设计意图注释（原文引用）

- `pi0.py:226-227` —— *"note that we use the convention more common in
  diffusion literature, where t=1 is noise and t=0 is the target
  distribution. yes, this is the opposite of the pi0 paper, and I'm sorry."*
  （即：代码里 t=1 是噪声、t=0 是目标分布，与 pi0 论文相反。）
- `gemma.py:444-450` —— §2.3 中引用的 expert 后缀命名原因。
- `README.md:8` —— §2.2c 中引用的 knowledge insulation 范围说明。
- `libero_policy.py:29-83` 的 docstring —— 明确写成「适配新机器人时可直接复制粘贴」的模板（见 §4.3）。

### 2.9 一页速查对比表

| 维度 | π0 | π0.5 | π0-FAST |
|---|---|---|---|
| 模型类 | `Pi0`（`pi05=False`） | `Pi0`（`pi05=True`） | `Pi0FAST`（独立类） |
| 动作头 | Flow matching（连续去噪，`sample_actions` = Euler 积分） | Flow matching（与 π0 完全相同的头） | 自回归 next-token 解码 |
| Loss | 速度场上的 MSE | 速度场上的 MSE | 动作 token 上的交叉熵 LM loss |
| Action expert 权重 | 独立的 `gemma_300m` mixture expert 分支 | 同 π0 | 无 —— 单个 LLM 包办一切 |
| State 输入 | 连续的 suffix token | 离散语言 token（state 被离散化进文本） | 离散语言 token（无条件） |
| 时间步注入 | Concat + MLP | AdaRMSNorm + 门控残差 | 不适用（没有扩散时间步） |
| Gemma 实现 | `gemma.py`（mixture-of-experts） | `gemma.py`（mixture-of-experts） | `gemma_fast.py`（单 expert，增量 KV cache） |
| Tokenizer | `PaligemmaTokenizer`（state=None） | `PaligemmaTokenizer`（state 离散化） | `FASTTokenizer`（FAST 压缩动作 token） |
| `max_token_len` 默认 | 48 | 200 | 250 |
| `action_horizon` 默认 | 50 | 50 | 32 |
| 推理时归一化策略 | 均值/标准差（z-score） | 分位数（q01/q99） | 分位数（q01/q99） |
| LoRA 支持 | 支持（两条分支均可） | 没有配置在用（机制存在） | 支持（单分支） |
| PyTorch 支持 | 支持 | 支持 | **不支持** |
| Knowledge insulation | 不适用 | 只见于论文/博客，**本仓库模型代码中没有** | 不适用 |

---

## 3. 训练流程

### 3.1 配置系统架构

核心文件：`src/openpi/training/config.py`（989 行）。

- **`DataConfig`**（`config.py:64-98`）—— 完全解析后的数据流水线描述：
  `repo_id`、`asset_id`、`norm_stats`、`repack_transforms` / `data_transforms` /
  `model_transforms`（各自是一个 `openpi.transforms.Group`）、`use_quantile_norm`、
  `action_sequence_keys`、`prompt_from_task`，以及仅 RLDS 用到的字段
  （`rlds_data_dir`、`action_space`、`datasets`）。
- **`DataConfigFactory`**（抽象类，`config.py:166-200`）—— 根据 model config 构建
  `DataConfig`。`create_base_config()`（`config.py:179-188`）解析 `asset_id`、
  加载 norm stats（`_load_norm_stats`, `config.py:190-200`，经
  `openpi.shared.normalize.load` + `openpi.shared.download.maybe_download`），
  并设置 `use_quantile_norm = model_type != ModelType.PI0`。
  具体子类：`FakeDataConfig`、`SimpleDataConfig`、`LeRobotAlohaDataConfig`、
  `LeRobotLiberoDataConfig`、`RLDSDroidDataConfig`、`LeRobotDROIDDataConfig`
  （`config.py:204-462`）。
- **`ModelTransformFactory`**（`config.py:106-163`）—— 按 `ModelType` 构建
  模型专属的输入/输出变换（选 tokenizer、图像 resize 到 224×224、state/action padding）。
- **`TrainConfig`**（frozen dataclass，`config.py:465-556`）—— 顶层配置：
  `model`、`weight_loader`、`pytorch_weight_path`、`pytorch_training_precision`
  （`"bfloat16"`/`"float32"`）、`lr_schedule`、`optimizer`、
  `ema_decay: float | None = 0.99`、`freeze_filter`、`data`、
  `assets_base_dir`/`checkpoint_base_dir`、`seed`、`batch_size`、`num_workers`、
  `num_train_steps`、`log_interval`、`save_interval`、`keep_period`、
  `overwrite`/`resume`、`wandb_enabled`、`policy_metadata`、
  `fsdp_devices: int = 1`（仅 JAX）。`trainable_filter` property
  （`config.py:549-552`）= `nnx.All(nnx.Param, nnx.Not(freeze_filter))`。
- 注册表：`_CONFIGS` 列表（`config.py:560-971`）→ `_CONFIGS_DICT`（`config.py:975`）；
  `get_config(name)`（`config.py:982-989`）会用 `difflib` 对拼写错误做模糊匹配提示。

### 3.2 已注册的训练配置（按模型 / 平台归类）

**π0（JAX）：** `pi0_aloha`、`pi0_aloha_towel`、`pi0_aloha_tupperware`、
`pi0_droid`、`pi0_libero`、`pi0_libero_low_mem_finetune`（LoRA）、
`pi0_aloha_pen_uncap`、`pi0_aloha_sim`。

**π0-FAST（JAX）：** `pi0_fast_droid`、`pi0_fast_libero`、
`pi0_fast_libero_low_mem_finetune`（LoRA）、`pi0_fast_full_droid_finetune`（RLDS）。

**π0.5（JAX，`Pi0Config(pi05=True)`）：** `pi05_aloha`、`pi05_droid`、
`pi05_libero`（还设了一个默认的 `pytorch_weight_path`）、`pi05_aloha_pen_uncap`、
`pi05_full_droid_finetune`（RLDS）、`pi05_droid_finetune`（LeRobot 格式的自定义 DROID 数据）。

**Debug：** `debug`、`debug_restore`、`debug_pi05`（用 `FakeDataConfig`）。

**额外的配置组**，在 `config.py:969-970` 处追加：
- `training/misc/roboarena_config.py` → `paligemma_binning_droid`、
  `paligemma_fast_droid`、`paligemma_fast_specialist_droid`、
  `paligemma_vq_droid`、`paligemma_diffusion_droid`。
- `training/misc/polaris_config.py` → `pi05_droid_jointpos_polaris`、
  `pi0_fast_droid_jointpos_polaris`、`pi0_droid_jointpos_polaris`、
  `pi0_droid_jointpos_100k_polaris`、`paligemma_binning_droid_jointpos`。

配置名唯一性在 import 时就会被强制校验（`config.py:973-974`）。

### 3.3 JAX 训练循环（`scripts/train.py`，280 行）

1. **初始化**（`main()`, `train.py:194-217`）：初始化日志；断言
   `batch_size % device_count == 0`；通过 `sharding.make_mesh(config.fsdp_devices)`
   构建设备 mesh（`sharding.py:17-23`，形状 `(device_count // fsdp_devices,
   fsdp_devices)`，轴名 `("batch", "fsdp")`）；初始化 Orbax `CheckpointManager`
   （item 为 `assets`、`train_state`、`params`）并检测是否需要 resume。
2. **W&B 初始化**（`train.py:218`）—— resume 时复用已保存的 run id，否则通过
   `dataclasses.asdict` 记录完整配置。
3. **Data loader**（`train.py:220-234`）—— `create_data_loader(config,
   sharding=data_sharding, shuffle=True)`；在 step 0 向 W&B 打一张拼接图做 sanity check。
4. **Train state 初始化**（`init_train_state`, `train.py:85-133`）—— 构建 optimizer、
   创建模型、通过 `nnx.split`/`state.replace_by_pure_dict`/`nnx.merge` 合入
   weight loader 提供的参数、把冻结参数转成 bf16、构建
   `TrainState(step, params, model_def, tx, opt_state, ema_decay, ema_params)`。
   逐数组的 sharding 由 `sharding.fsdp_sharding` 推导（`sharding.py:48-102`：
   对 ≥4MiB 且 ≥2 维的数组，沿最大的可整除轴切分，其余全复制）。
5. **Resume**（`train.py:240-241`）—— 需要时用 `_checkpoints.restore_state` 从 Orbax 恢复。
6. **JIT 编译训练步**（`train.py:243-248`）。
7. **主循环**（`train.py:258-273`）—— 每步执行 `train_step`；每 `log_interval` 步
   对指标取均值并记录；每 `save_interval` 步（以及最后一步）调用
   `_checkpoints.save_state`。
8. **`train_step`**（`train.py:137-191`）—— 把 `model_def`+`params` merge 成 nnx 模型；
   调 `model.compute_loss(rng, obs, actions, train=True)`；只对 `trainable_filter`
   选中的参数求梯度（`nnx.DiffState`/`nnx.value_and_grad`）；optimizer 更新 +
   `nnx.update`；若启用 EMA 则做 `ema_params = decay*old + (1-decay)*new`
   （`train.py:169-175`）；返回 `loss`、`grad_norm`、`param_norm`。

### 3.4 PyTorch 训练循环（`scripts/train_pytorch.py`，632 行）—— 与 JAX 版已核实的差距

- **分布式策略**：普通的 `torch.nn.parallel.DistributedDataParallel`
  （`setup_ddp`, `train_pytorch.py:94-109`）—— **没有 FSDP**
  （已确认：全文零处 FSDP 引用）。
- **Checkpoint**：自定义格式，不用 Orbax —— `model.safetensors`
  （`safetensors.torch.save_model`）、`optimizer.pt`（裸 `torch.save`）、
  `metadata.pt`，并通过临时目录原子重命名落盘（`save_checkpoint`,
  `train_pytorch.py:149-194`）。**没有 EMA** —— 代码里直接打日志
  `"EMA is not supported for PyTorch training"`（`train_pytorch.py:499`）；
  `TrainState` 里 `ema_params`/`ema_decay` 那套概念从未被触碰。
- **LoRA**：本文件从未读取 `freeze_filter`/`trainable_filter`；
  `gemma_pytorch.py`/`pi0_pytorch.py` 里也完全没有 `"lora"` 字符串 —— **未实现**。
- **Optimizer / 学习率**：只有 `torch.optim.AdamW`，直接从
  `config.optimizer.b1/b2/eps/weight_decay` 构建（`train_pytorch.py:458-464`）；
  完全绕过了 `optimizer.py` 里那套基于 `optax` 的机制；学习率调度是手写复现的，
  模仿 `CosineDecaySchedule`（`lr_schedule()`, `train_pytorch.py:472-480`），逐步手动施加。
- **精度**：只有全局 bf16-或-fp32 一个开关（`pytorch_training_precision`）
  —— 不是 JAX 那套真正的混合精度方案。
- **数据加载**：和 JAX 复用同一个 `_data_loader.create_data_loader(config,
  framework="pytorch", ...)`，只是 DDP 下用 `DistributedSampler` 取代 JAX sharding。
  **RLDS/DROID 加载会显式抛 `NotImplementedError`**（`data_loader.py:367-368`）
  —— 也就是说所有 DROID RLDS 配置（`pi0_fast_full_droid_finetune`、
  `pi05_full_droid_finetune`、polaris/roboarena 的 DROID 配置）都没法用
  `train_pytorch.py` 跑。
- **权重加载**：走完全独立的路径 —— `config.pytorch_weight_path` 指向一个含
  `model.safetensors` 的目录，用 `safetensors.torch.load_model` 加载
  （`train_pytorch.py:442-449`）；JAX 的 `WeightLoader` 抽象完全没被用到。
- 若模型暴露了 `gradient_checkpointing_enable`，则支持 gradient checkpointing
  （`train_pytorch.py:411-417`）。

### 3.5 LoRA 接线细节

- `gemma.py:55`：`Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora",
  "gemma_2b", "gemma_2b_lora"]`；`get_config()`（`gemma.py:58-107`）在 variant 名
  含 `_lora` 时挂上 `lora_configs`。
- `Pi0Config.get_freeze_filter()`（`pi0_config.py:88-117`）：
  当 `paligemma_variant` 含 `"lora"` 时冻结 `.*llm.*`；
  当 `action_expert_variant` 含 `"lora"` 时冻结 `.*llm.*_1.*`（action expert）；
  两种情况下都始终排除 `.*lora.*`，以保证 adapter 可训练。
- `Pi0FASTConfig.get_freeze_filter()`（`pi0_fast.py:127-130`）：同思路，只有单个 LLM。
- 接入示例：`pi0_libero_low_mem_finetune`（`config.py:693-695`，
  `paligemma_variant="gemma_2b_lora"`、`action_expert_variant="gemma_300m_lora"`、
  `ema_decay=None`）与 `pi0_fast_libero_low_mem_finetune`（`config.py:737-739`）。
- `weight_loaders.CheckpointWeightLoader.load()`（`weight_loaders.py:50-54`）
  会把缺失的 `.*lora.*` 参数从全新初始化的参数里补入，因为 base checkpoint 不含 LoRA 权重。

### 3.6 数据加载流水线

在 `create_data_loader`（`data_loader.py:223-268`）里按 `data_config.rlds_data_dir`
是否设置来分派到两套后端：

- **LeRobot 数据集路径**（`create_torch_dataset`, `data_loader.py:130-151`）——
  包装 `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`；
  按 `action_sequence_keys` 依 `action_horizon / fps` 计算 `delta_timestamps`
  以取出 action chunk；批处理由 `TorchDataLoader`（`data_loader.py:381-468`）完成
  —— 它是 `torch.utils.data.DataLoader` 的薄封装，带一个 numpy 堆叠的
  `_collate_fn`，以及一个在 worker 中关闭 XLA GPU 预分配的 `_worker_init_fn`。
- **DROID RLDS 路径**（`droid_rlds_dataset.py`，`DroidRldsDataset`，36-248 行）——
  基于 `dlimp`/`tensorflow_datasets`：从 `tfds.builder` 构建，按文件名正则过滤出
  成功轨迹，`.repeat()`，可选的逐 episode 静止帧过滤表（由 JSON 驱动），
  `restructure()`（重映射 key，从 2 个外部相机中随机选 1 个、从 3 条语言指令中随机选 1 条），
  `chunk_actions()`（带边界钳制的 action chunk gather），展平为逐 step，
  施加静止/成功过滤，最后惰性解码 JPEG。多个 RLDS 数据集通过
  `dl.DLataset.sample_from_datasets(weights=...)` 混合（权重必须和为 1.0）。
  **批处理是在 TF 内部完成的**，所以 `RLDSDataLoader`（`data_loader.py:486-527`）
  只是一个透传的 sharding 封装。`__len__` 是硬编码的近似常量 `20_000_000`。
  要求 `num_workers=0`（TF/dlimp 内部自己管多进程）。
- **两条路径共用的变换流水线**（`data_loader.py:172-220`）：
  `repack_transforms.inputs` → `data_transforms.inputs` →
  `Normalize(norm_stats, use_quantiles=...)` → `model_transforms.inputs`。
  若 `norm_stats is None` 会报错（除非 `repo_id == "fake"` 或显式 `skip_norm_stats`）。
- **归一化统计量的计算**（`scripts/compute_norm_stats.py`）——
  用同一套数据集，但只走到 `repack_transforms` + `data_transforms`（归一化之前）；
  对 `"state"`/`"actions"` 累积 `normalize.RunningStats()`；
  通过 `normalize.save()` 写到 `config.assets_dirs / data_config.repo_id`。

### 3.7 Checkpoint 格式与 EMA（JAX）

- **基于 Orbax**（`checkpoints.py`）—— `CheckpointManager` 管 3 个 item：
  `"assets"`（任意异步副作用式保存，如 norm stats）、`"train_state"`、`"params"`
  （后两者用 `PyTreeCheckpointHandler`）；`max_to_keep=1`，
  `keep_period` 可配置以永久保留特定 step。
- **保存逻辑**（`checkpoints.py:65-86`）：**若启用 EMA**，`"params"` item 存的是
  `ema_params`（可直接用于推理），而 `train_state.params` 被清空；
  **若未启用 EMA**，`"params"` 存原始训练参数，而 `train_state.params` 被清空。
  这种双 item 拆分使得可以只用一个轻量的 params-only checkpoint 做推理服务，
  不必携带 optimizer state。
- **EMA 更新**在 `train.py` 的 `train_step` 中（`train.py:169-175`）：
  `ema_params = ema_decay*old_ema + (1-ema_decay)*new`。
  默认 `ema_decay=0.99`；`pi05_libero` 覆盖为 `0.999`（`config.py:759`）；
  LoRA 配置则设 `ema_decay=None` 关闭它。
- **PyTorch 的 checkpoint 结构完全不同**（见 §3.4）—— 纯 `safetensors`/`torch.save`，
  无 Orbax、无 EMA、也没有 params-only 的推理专用拆分（模型和 optimizer state 总是一起存）。

---

## 4. 数据 / Policy 层

### 4.1 `Policy.infer()` 全链路（`src/openpi/policies/policy.py`）

1. **拷贝** 原始 obs dict（`policy.py:70`），避免变换污染调用方数据。
2. **输入变换**（`policy.py:71`）：组合链 = repack → 机器人 `*Inputs` →
   `Normalize` → 模型变换（tokenization、resize、padding）。
3. **加 batch 维 / 放置到设备**（`policy.py:72-79`）：JAX —— 增加 batch 轴、
   `jnp.asarray`、split RNG；PyTorch —— `torch.from_numpy(...).to(device)`。
4. **可选的外部噪声** 接线（`policy.py:82-88`）。
5. **构建 `Observation`**：`Observation.from_dict(inputs)`
   （`policy.py:90`；`model.py:83-136`）—— 把 uint8 图像缩放为 float32 的
   `[-1, 1]`（`model.py:116-120`）。
6. **模型调用**：`model.sample_actions(rng_or_device, observation,
   **sample_kwargs)`（`policy.py:92-95`），用 `time.monotonic()` 计时。
7. **去 batch 维**，剥掉前导维并转 numpy（`policy.py:97-100`）。
8. **输出变换**（`policy.py:102`）：模型变换（如 FAST 反 tokenize）→
   `Unnormalize` → 机器人 `*Outputs` → repack 输出。
9. 附加 `outputs["policy_timing"] = {"infer_ms": ...}` 并返回。

`PolicyRecorder`（`policy.py:113-135`）可包装任意 `BasePolicy`，
把 `{"inputs": obs, "outputs": results}`（展平后）逐步 dump 成 `.npy` 文件
用于离线调试 —— 通过 `serve_policy.py --record` 启用。

### 4.2 `create_trained_policy` 的接线（`src/openpi/policies/policy_config.py:16-94`）

1. `checkpoint_dir = download.maybe_download(...)`（第 46 行）。
2. **PyTorch vs JAX 自动识别**：检查 checkpoint 目录里是否存在
   `model.safetensors`（第 49-50 行）。
3. `data_config = train_config.data.create(...)`（第 58 行）——
   解析出机器人专属的 `DataConfig`（如 `LeRobotAlohaDataConfig`）。
4. **Norm stats**：从 **checkpoint 自己的** `assets/<asset_id>` 目录加载
   （第 59-64 行）—— *而不是*训练配置里的 `assets_dir` ——
   目的是确保推理用的统计量与训练时完全一致（第 60-61 行注释有说明）。
5. PyTorch 设备自动解析为有 CUDA 则 `"cuda"`、否则 `"cpu"`（第 66-73 行）。
6. 组装完整的变换流水线（第 75-94 行），顺序与 §4.1 一致。

### 4.3 各平台变换对比

三个文件（`aloha_policy.py`、`droid_policy.py`、`libero_policy.py`）
都实现同样的 `Inputs`/`Outputs` 模式，最终都汇入模型通用的
「3 张图 + 单个 state 向量」的 `Observation` schema。

| | `AlohaInputs`/`AlohaOutputs` | `DroidInputs`/`DroidOutputs` | `LiberoInputs`/`LiberoOutputs` |
|---|---|---|---|
| 原始 state 维度 | 14（双臂：2×(6 关节+1 夹爪)） | 8（7 关节 + 1 夹爪） | 8（7 自由度 + 夹爪，Panda 仿真） |
| 输出动作维度 | 14 | 8 | 7 |
| 用到的 image key | `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`（第三个外部视角没有映射进去） | 依模型类型而定 —— 见下 | `base_0_rgb`=image、`left_wrist_0_rgb`=wrist_image、`right_wrist_0_rgb`=zeros |
| 特殊处理 | 关节符号翻转 mask + 夹爪线性↔角度换算，用的是 Interbotix/Aloha 的标定常数（`_gripper_to_angular`, `aloha_policy.py:117-137`） | 依模型类型的 image 槽位路由（见下） | docstring 明确写成「新机器人适配模板」 |

**DROID 依模型类型的 image 槽位路由**（`droid_policy.py:47-58`）是最值得注意的特例：
- PI0/PI05：`names=(base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)`，
  images=`(base, wrist, zeros_like(base))`，masks=`(True, True, False)`
  —— 右手腕位被 mask 掉当作 padding。
- PI0_FAST：`names=(base_0_rgb, base_1_rgb, wrist_0_rgb)`，
  images=`(base, zeros_like(base), wrist)`，masks=`(True, True, True)`
  —— **FAST 从不 mask padding 图像**（三张都被当作有效输入 token），
  而且它是把零填进*第二个 base 槽位*，而非沿用 flow matching 模型的手腕槽位约定。

同样这条「flow matching 模型 mask padding 图像、FAST 不 mask」的规律，
也出现在 `libero_policy.py:68`（`image_mask["right_wrist_0_rgb"]`
只有在 `PI0_FAST` 下才是 `True`）。

### 4.4 归一化 —— 分位数 vs z-score

由 `use_quantiles` 控制（`Normalize`/`Unnormalize` 在
`src/openpi/transforms.py:114-181`），其值来自
`use_quantile_norm = model_type != ModelType.PI0`（`training/config.py:187`）。

| | 均值/标准差（π0） | 分位数 q01/q99（π0.5、π0-FAST） |
|---|---|---|
| Normalize | `(x - mean) / (std + 1e-6)`（`transforms.py:137-139`） | `(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0`（`transforms.py:141-145`） |
| Unnormalize | `x * (std + 1e-6) + mean`（`transforms.py:170-173`） | `(x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01`（`transforms.py:175-181`） |

统计量由 `src/openpi/shared/normalize.py` 的 `RunningStats` 计算
（流式累积均值/平方均值 → 标准差，外加每维 5000 个 bin 的直方图用于估计 `q01`/`q99`）。
`get_statistics()` **始终填满四个字段**（mean、std、q01、q99）；
具体*使用*哪一对，是在归一化时由 `use_quantiles` flag 决定的，
并不在统计量采集阶段固化。最终以 `norm_stats.json` 通过 `save`/`load` 持久化。

### 4.5 Checkpoint 下载与缓存（`src/openpi/shared/download.py`）

- 缓存根目录：环境变量 `OPENPI_DATA_HOME`，默认 `~/.cache/openpi`。
- `maybe_download(url)`：本地路径直接短路返回；远程路径映射到
  `cache_dir / netloc / path`；用 `filelock.FileLock` 保证并发安全；
  先下载到 `.partial` 临时路径，再原子性 move 到位。
- **`gs://openpi-assets/...` 特意走 `gsutil` 命令行**，
  以规避该 bucket 上的 `gcsfs` 认证问题；若未安装 `gsutil` 则回退到
  `fsspec`/`gcsfs`；其他 URL 则直接用 `fsspec`/`gcsfs` 并带 `tqdm` 进度条。
- 下载后会统一规范化权限，使得共享缓存在「容器内 + 宿主机脚本」两种用户下都能用。
- **缓存失效策略**：若缓存文件的 mtime 早于某个硬编码的过期日期，
  特定路径前缀（如 `pi0_aloha_pen_uncap`、`pi0_libero`，以及通用的
  `openpi-assets/checkpoints/` 前缀）会被强制失效 ——
  以便上游发布新 checkpoint 后强制重新下载。

### 4.6 `nnx_utils.py`

- `module_jit(meth, ...)`（`nnx_utils.py:15-43`）—— 把绑定的 `nnx.Module`
  方法 jit 编译：先拆成 `(graphdef, state)`，再对闭包住冻结 state 的普通函数做 jit，
  从而避开 `nnx.jit` 额外的图追踪开销。`Policy.__init__` 用它来 jit
  `model.sample_actions`。
- `PathRegex`（`nnx_utils.py:46-63`）—— 一个按正则匹配参数路径的 `nnx` filter
  （用于 LoRA 选择和 bf16 转换）。
- `state_map`（`nnx_utils.py:66-69`）—— 对被 filter 选中的 `nnx.State` 叶子施加函数。

---

## 5. 推理服务与远程推理基础设施

### 5.1 整体架构

两个进程通过**单条持久 WebSocket 连接**通信：

- **服务端**（GPU 机器）：`src/openpi/serving/websocket_policy_server.py`
  —— 包装一个 `Policy` 并通过 websocket 暴露出去。
- **客户端**（机器人机器）：`packages/openpi-client/src/openpi_client/
  websocket_client_policy.py` —— 依赖极少（websockets + msgpack + numpy），
  设计上就是为了直接 pip 装进机器人代码库。

**协议**（`websocket_policy_server.py:34-91`）：
1. 服务端绑定 `0.0.0.0:<port>`（默认 8000），`compression=None`、
   `max_size=None`（因为图像负载很大），并在 websocket 握手之前就用
   `200 OK` 响应纯 HTTP 的 `GET /healthz`（给 docker/k8s 探针用）。
2. 连接建立后，服务端立刻发送一个 msgpack 打包的 **metadata dict**
   （`policy.metadata`）—— 这是客户端读到的第一帧。
3. 循环：`recv()` msgpack 的 obs dict → `policy.infer(obs)` → 打包结果 → `send()`。
   并在返回的 dict 里塞进 `server_timing`（`infer_ms`，以及
   `prev_total_ms` —— 度量的是*上一步*的往返 + 发送耗时）。
4. 错误处理：捕获异常，把完整 traceback 作为**文本帧**（而非 msgpack 字节）发出，
   然后以 `INTERNAL_ERROR` 关闭连接。客户端通过 `isinstance(response, str)`
   检测到并抛 `RuntimeError`。
5. 客户端（`websocket_client_policy.py`）用的是**同步**的
   `websockets.sync.client`（匹配机器人控制循环通常是同步的这一事实）；
   服务端用 `websockets.asyncio`（`serve_forever` 需要）。
   客户端会每 5 秒重试连接直到服务端起来 —— 这对 docker-compose 启动竞态很友好。

不使用 TLS（只有 `ws://`）；可选 `Authorization: Api-Key` 头做认证。

### 5.2 `scripts/serve_policy.py` —— CLI 与启动流程

通过 `tyro.cli(Args)` 提供两种模式：
- **默认模式**：`--env {aloha|aloha_sim|droid|libero}` 选中一个硬编码的
  `DEFAULT_CHECKPOINT` 条目（例如 `droid` →
  `config="pi05_droid"`、`dir="gs://openpi-assets/checkpoints/pi05_droid"`）。
- **`policy:checkpoint` 子命令**：`--policy.config=<name>
  --policy.dir=<路径或 gs:// URI>`，可用于任意已注册的 `TrainConfig`。

启动流程（`serve_policy.py:99-117`）：
1. `create_policy(args)` → `create_trained_policy(...)`（见 §4.2）。
2. 取出 `policy.metadata` —— 这就是连接建立时服务端要发给客户端的内容。
3. 若指定 `--record`，包装成 `PolicyRecorder`。
4. 打印 hostname/IP（仅供参考 —— 服务端仍然绑定 `0.0.0.0`）。
5. 构造 `WebsocketPolicyServer(...)` 并调用 `serve_forever()`（阻塞）。

### 5.3 `action_chunk_broker.py` —— 动作块分发的确切逻辑

文件：`packages/openpi-client/src/openpi_client/action_chunk_broker.py`（50 行）。
它是**客户端侧**对另一个 `BasePolicy`（通常是 `WebsocketClientPolicy`）的装饰器
—— 服务端完全不用它。使用处：`examples/aloha_sim/main.py:36-42`、
`examples/aloha_real/main.py`。

`infer(obs)` 的逻辑：
1. `if self._last_results is None:` → 恰好调用内层 policy 一次（一次网络往返），
   缓存完整结果，重置 `_cur_step = 0`。**这是唯一会发起新服务端查询的地方。**
2. 否则复用缓存结果 —— **不发生网络调用**，
   而且本次 tick 新传进来的 `obs` 会被**静默忽略**。
3. 从缓存结果的每个数组叶子里切出第 `_cur_step` 个时间步
   （`tree.map_structure`）；非数组叶子原样透传。
4. `_cur_step += 1`。
5. 若 `_cur_step >= action_horizon`，清空 `_last_results` —— 下次调用即触发新查询。

**净效果：服务端恰好每 `action_horizon` 次 `infer()` 调用才被查询一次**，
而不是按墙上时钟定时。两次查询之间是完全**开环**执行的 ——
机器人不管当前观测长什么样，都照着预先算好的 chunk 执行下去。
这与 `docs/remote_inference.md:60-64` 的建议一致：每 N 步查一次 policy，
其余步开环执行 chunk 剩余部分。

> 读者注意：控制循环每个 tick 仍然会调用 `broker.infer(obs)` ——
> broker 在缓存命中的那些步会静默丢弃 `obs`。
> 不要以为每次 `infer()` 都会打到服务端。

### 5.4 序列化：msgpack + numpy

`packages/openpi-client/src/openpi_client/msgpack_numpy.py` —— 改编自
`lebedov/msgpack-numpy`，重写的目的是**避免 object 数组回退到 pickle**
（安全考虑：反序列化网络数据时避免任意代码执行）。
`pack_array`/`unpack_array` 把 `np.ndarray` 序列化为「裸字节 + dtype + shape」
（不压缩）；拒绝 void/object/complex 这几类 dtype。
选它的理由：无需 schema（适合各机器人形态各异的临时 observation dict）、
速度快（按该模块自带的 benchmark 注释，大数组比 pickle 快约 4 倍）、
以及跨语言兼容性。服务端和客户端都在一条连接上复用同一个长生命周期的 `Packer()`。

### 5.5 Docker 环境（`docs/docker.md`）

推荐所有示例都用 Docker（避免在宿主机上装 ROS/机器人依赖）。
需要 rootless Docker + NVIDIA Container Toolkit。明确列出的不兼容项：
通过 snap 安装的 Docker 会导致无法访问 `libnvidia-ml.so`；
Docker Desktop 与 NVIDIA runtime 不兼容。便捷脚本：
`scripts/docker/install_docker_ubuntu22.sh`、
`scripts/docker/install_nvidia_container_toolkit.sh`。用法：
`docker compose -f scripts/docker/compose.yml up --build`（顶层）或
`docker compose -f examples/<name>/compose.yml up --build`（单个示例）。

### 5.6 Norm stats 复用机制（`docs/norm_stats.md`）

当你要微调的机器人与某个预训练平台相似时，可以**复用预训练的 norm stats**
而不是重新计算，这样新动作在归一化空间里对模型来说会更"熟悉"。

- 在 `TrainConfig.data` 的 factory 上加
  `AssetsConfig(assets_dir=..., asset_id=...)`，可覆盖**训练**时统计量的加载来源
  （`training/config.py:38-58`、`179-199`）。
- **推理**时，`create_trained_policy` 总是从 **checkpoint 自己的**
  `assets/<asset_id>` 重新加载（见 §4.2 第 4 步）——
  也就是说，无论训练时选了哪种复用方式，checkpoint 在 norm stats 上都是自包含的。
- 支持的预训练机器人 ID（`docs/norm_stats.md:34-44`）：
  `trossen`（ALOHA）、`trossen_mobile`（Mobile ALOHA）、`droid`（DROID Franka）、
  `franka`（非 DROID 的 Franka）、`ur5e`、`ur5e_dual`、`arx`、
  `arx_mobile`、`fibocom_mobile`。
- 仅当你的动作空间约定与文档记载的预训练动作空间一致时才有意义
  （关节角布局、夹爪范围 `[0,1]`、各机器人的控制频率 ——
  例如 DROID：关节速度维 0-6 + 夹爪维 7 @ 15 Hz）。
  文档建议两种方式（复用 vs 重新计算）都实测一下，取效果好的那个。

### 5.7 `examples/simple_client` —— 无机器人也能测

一个极简客户端（`main.py`），发送按 `aloha`/`droid`/`libero`
各自期望 schema 构造的 **合成**（`np.random`）观测，
并报告推理延迟（通过 `TimingRecorder` 给出均值/标准差/分位数，可选导出 parquet）。
无机器人冒烟测试的标准做法：

```
# 终端 1
uv run scripts/serve_policy.py --env DROID
# 终端 2
uv run examples/simple_client/main.py --env DROID
```

计时开始前会先调两次 `policy.infer()` 做预热（触发 JIT / 模型加载）。
也可以用 `docker compose -f examples/simple_client/compose.yml up --build` 跑。

---

## 6. 示例与第三方集成

| 示例 | 平台 | 关键模式 |
|---|---|---|
| `examples/aloha_real` | 真实双臂 ALOHA（通过 `third_party/aloha` 这个带 Realsense 支持的 ROS fork） | 3 个进程：机器人控制循环、`roslaunch aloha ros_nodes.launch`、policy server。附带 `pi0_aloha_towel`/`tupperware`/`pen_uncap` checkpoint；微调可用 `convert_aloha_data_to_lerobot.py` + 复用 `trossen` 的 norm stats |
| `examples/aloha_sim` | MuJoCo 仿真 ALOHA（transfer-cube 任务） | 最简单的示例 —— `MUJOCO_GL=egl python examples/aloha_sim/main.py` + `serve_policy.py --env ALOHA_SIM` |
| `examples/droid` | 真实 DROID（Franka） | 控制笔记本无 GPU → **远程** server 模式；还记录了 RoboArena 的 baseline checkpoint（FAST、binning、FSQ、diffusion 等变体） |
| `examples/libero` | LIBERO 仿真 benchmark（`third_party/libero`） | 推荐用 Docker Compose（需配 X11/EGL）；给出了 π0.5 @ 30k 的 benchmark 结果（Spatial/Object/Goal/10 四个 suite 平均约 96.85%）；内含 `convert_libero_data_to_lerobot.py` |
| `examples/ur5` | 不可直接运行 —— 是「新增机器人」的**教程模板** | 定义 `UR5Inputs`/`UR5Outputs` + `LeRobotUR5DataConfig`，是「接入自己机器人」的标准范例 |
| `examples/simple_client` | 无机器人 | 合成观测的延迟测试客户端（见 §5.7） |

`third_party/aloha`（submodule → `Physical-Intelligence/aloha`，
是 `tonyzhaozh/aloha` 的 fork）：ROS 包/launch 文件、硬件脚本
（`realsense_publisher.py`）、`aloha2` 变体。

`third_party/libero`（submodule → `Lifelong-Robot-Learning/LIBERO`）：
LIBERO benchmark 的 MuJoCo 环境与任务/benchmark 定义。

`examples/libero/convert_libero_data_to_lerobot.py` 是 LeRobot 数据集转换的标准模板：
读入 4 个原始 Libero RLDS/TFDS split，写出一个 `LeRobotDataset`，
schema 为 `image`/`wrist_image`（256×256×3）、`state`（8,）、`actions`（7,）
—— 与 `LiberoInputs`/`LiberoOutputs` 精确对应。注释写明：
*"OpenPi assumes that proprio is stored in `state` and actions in `action`."*
Aloha/DROID 的转换脚本沿用/改编自它。

---

## 7. 文件速查索引

| 文件 | 内容 |
|---|---|
| `src/openpi/models/model.py` | `BaseModel` 接口、`Observation`/`Actions` dataclass、`ModelType` 枚举 |
| `src/openpi/models/pi0_config.py` | `Pi0Config`（π0/π0.5 共用配置）、`pi05` flag、freeze filter |
| `src/openpi/models/pi0.py` | `Pi0` 类 —— π0 与 π0.5 共用的 flow matching 模型 |
| `src/openpi/models/pi0_fast.py` | `Pi0FASTConfig`、`Pi0FAST` —— 自回归模型 |
| `src/openpi/models/gemma.py` | Mixture-of-experts Gemma + AdaRMSNorm（π0/π0.5） |
| `src/openpi/models/gemma_fast.py` | 单 expert Gemma + 增量 KV cache（π0-FAST） |
| `src/openpi/models/siglip.py` | 视觉编码器（三个模型共用） |
| `src/openpi/models/tokenizer.py` | `PaligemmaTokenizer` vs `FASTTokenizer` |
| `src/openpi/models/lora.py` | LoRA adapter 模块 |
| `src/openpi/models_pytorch/pi0_pytorch.py`、`gemma_pytorch.py` | PyTorch 复现版（仅 π0/π0.5） |
| `src/openpi/training/config.py` | `TrainConfig`/`DataConfig` 注册表 —— 整个系统的中枢 |
| `src/openpi/training/weight_loaders.py` | 微调用的 checkpoint 权重加载 |
| `src/openpi/training/checkpoints.py` | Orbax checkpoint 存取、EMA 拆分 |
| `src/openpi/training/data_loader.py`、`droid_rlds_dataset.py` | LeRobot vs DROID RLDS 两套数据流水线 |
| `scripts/train.py` / `train_pytorch.py` | JAX / PyTorch 训练入口 |
| `scripts/serve_policy.py` | Policy server 命令行入口 |
| `scripts/compute_norm_stats.py` | 归一化统计量预计算 |
| `src/openpi/policies/policy.py`、`policy_config.py` | `Policy.infer()`、`create_trained_policy` |
| `src/openpi/policies/aloha_policy.py`、`droid_policy.py`、`libero_policy.py` | 各机器人的输入/输出变换 |
| `src/openpi/shared/normalize.py` | `RunningStats`、分位数/z-score 归一化 |
| `src/openpi/shared/download.py` | Checkpoint 下载与缓存（`OPENPI_DATA_HOME`） |
| `src/openpi/serving/websocket_policy_server.py` | 服务端 websocket policy 循环 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | 客户端 websocket 连接 |
| `packages/openpi-client/src/openpi_client/action_chunk_broker.py` | 开环动作块分发 |
| `packages/openpi-client/src/openpi_client/msgpack_numpy.py` | 线上传输的序列化格式 |

---

## 8. 功能清单 —— 这个仓库到底能做什么

前面几章是按「代码怎么写的」组织的深度剖析；这一章换个角度，按「能用来做什么」
重新梳理一遍仓库提供的能力，方便做任务规划时直接查表。每条都指回前面对应的
深度分析章节，细节不重复展开。

### 8.1 模型与 Checkpoint

- **三种可直接使用的 base checkpoint**（README.md:61-68）：
  `pi0_base`、`pi0_fast_base`、`pi05_base`，均在 10k+ 小时机器人数据上预训练，
  可直接微调到自己的机器人/任务上。
- **七个针对具体平台/任务微调好的 "expert" checkpoint**（README.md:70-82）：
  `pi0_fast_droid`、`pi0_droid`、`pi0_aloha_towel`、`pi0_aloha_tupperware`、
  `pi0_aloha_pen_uncap`、`pi05_libero`、`pi05_droid` —— 部分可以直接 0-shot
  跑在对应机器人上（如 DROID、ALOHA towel），不需要再训练。
- **Checkpoint 自动下载与缓存**（§4.5）：默认从 `gs://openpi-assets` 拉取，
  缓存到 `~/.cache/openpi`，可用环境变量 `OPENPI_DATA_HOME` 改路径（README.md:84）。
- 三种模型架构的能力边界见 §2.9 速查表（谁支持 LoRA、谁支持 PyTorch 等）。

### 8.2 训练与微调能力

| 能力 | JAX（`train.py`） | PyTorch（`train_pytorch.py`） |
|---|---|---|
| 全量微调 | 支持 | 支持 |
| LoRA 微调 | 支持（§2.5、§3.5） | **不支持**（README.md:197） |
| 分布式训练 | FSDP，单机多卡，`fsdp_devices` 配置项（README.md:24）；**不支持多机**（同一行注明） | DDP，单机多卡 **和多机多卡**，用 `torchrun`（README.md:277-294：`--standalone` 单机 / `--node_rank`+`--master_addr` 多机） |
| EMA | 支持，默认 `ema_decay=0.99`（§3.7） | **不支持**（`train_pytorch.py:499` 直接打日志说明） |
| 混合精度 | 真正的混合精度（权重/梯度 fp32，激活 bf16），`dtype` 可切 float32 | 只有整体 bf16 或整体 fp32 二选一（`pytorch_training_precision`），不支持真混合精度 |
| Gradient checkpointing | — | 支持，模型若暴露 `gradient_checkpointing_enable` 即自动启用（`train_pytorch.py:411-417`） |
| Resume | 支持（Orbax） | 支持（`--resume`，safetensors） |
| 快速验证 | `debug`/`debug_restore`/`debug_pi05` 配置（`FakeDataConfig`，不需要真实数据即可跑通整条训练链路） | 同上，`uv run scripts/train_pytorch.py debug --exp_name pytorch_test` |

**显存门槛参考**（README.md:28-32，单卡估算，可配 `fsdp_devices` 做模型并行分摊）：

| 模式 | 所需显存 | 参考 GPU |
|---|---|---|
| 推理 | > 8 GB | RTX 4090 |
| LoRA 微调 | > 22.5 GB | RTX 4090 |
| 全量微调 | > 70 GB | A100(80GB) / H100 |

训练配置系统本身的能力（配置注册表、模糊匹配报错等）见 §3.1。

### 8.3 数据处理与格式转换

- **两套数据加载后端**：LeRobot 数据集 / DROID RLDS（仅 JAX，PyTorch 侧
  `data_loader.py:367-368` 显式抛 `NotImplementedError`）—— 细节见 §3.6。
- **「自己机器人数据 → LeRobot 数据集」转换脚本模板**：
  `examples/libero/convert_libero_data_to_lerobot.py`（README.md:131-139，§6）
  —— README 明确说明这是一个可以直接照抄改造的最小示例，ALOHA/DROID 的转换脚本
  也是照这个模式写的。
- **JAX ↔ PyTorch checkpoint 互转**：`examples/convert_jax_model_to_pytorch.py`
  （README.md:214-223、254-263）。除了做权重格式转换，还支持
  `--inspect_only` 模式，只打印 checkpoint 里全部参数的层级路径而不做转换
  （`convert_jax_model_to_pytorch.py:5-16`）——排查参数命名/shape 不对齐问题时很好用。
- **归一化统计量计算**：`scripts/compute_norm_stats.py`（§3.6），
  训练前必须先跑，否则 `Normalize` 变换会直接报错。
- **跨机器人复用预训练 norm stats**：无需重新计算统计量即可微调到预训练
  覆盖过的相似机器人上（trossen/droid/franka/ur5e 等，§5.6）。

### 8.4 推理与部署

- **本地 Python API**：几行代码加载 checkpoint 并调用 `policy.infer()`
  （README.md:91-111，§4.1、§4.2），JAX/PyTorch checkpoint 用同一套 API，
  会自动探测格式（§4.2 第 2 步）。
- **WebSocket 远程推理服务**（§5）：GPU 机器跑 `serve_policy.py` 起服务，
  机器人侧用轻量 `openpi-client`（仅依赖 websockets+msgpack+numpy）连接；
  自带 `/healthz` HTTP 探活接口（`websocket_policy_server.py:87`）和可选的
  `Authorization: Api-Key` 认证头（§5.1）。这让 GPU 推理和机器人控制环境
  彼此解耦，可以用更强的 GPU 在机器人之外跑模型。
- **开环动作块分发**（`action_chunk_broker.py`，§5.3）：每
  `action_horizon` 步才真正查询一次服务端，中间步骤复用缓存结果——
  降低查询频率、容忍网络延迟。
- **推理过程录制与离线分析**：`PolicyRecorder`（`--record`，§4.1）把每步
  inputs/outputs 落盘成 `.npy`；配套的 `examples/policy_records.ipynb`
  读取这些记录，把输入输出的关节角曲线画出来对比，用于排查微调后策略的异常行为。
- **无机器人冒烟测试**：`examples/simple_client`（§5.7）生成合成观测，
  在没有真实机器人的情况下验证服务是否正常、测延迟。
- **Docker / Docker Compose 一键部署**（§5.5）：`scripts/docker/compose.yml`
  + `serve_policy.Dockerfile`，并提供 `install_docker_ubuntu22.sh`、
  `install_nvidia_container_toolkit.sh` 两个环境安装脚本。

### 8.5 多机器人平台集成（示例）

详见 §6 的完整表格；按能力简述：

- **ALOHA**：真实双臂（`examples/aloha_real`，经 ROS fork 驱动）与
  MuJoCo 仿真（`examples/aloha_sim`）两种形态都有完整示例。
- **DROID**（Franka）：真实机器人示例，支持远程 server 模式
  （控制端笔记本无 GPU 时很有用），并额外收录了 RoboArena 的多种 baseline
  checkpoint（FAST / binning / FSQ / diffusion 等）用于研究对比——
  这些 baseline **不是** pi0 系模型，只是同一套 benchmark 下的对照组。
- **LIBERO**：仿真 benchmark，提供 Docker Compose 一体化工作流
  （policy server + 评测脚本一起起），README 给出了 π0.5 在四个 suite 上
  约 96.85% 的 benchmark 结果。
- **UR5**：不是可直接运行的示例，而是「接入一个全新机器人」的标准教程模板
  （`UR5Inputs`/`UR5Outputs` + `LeRobotUR5DataConfig`）。

### 8.6 Notebook / 可视化调试工具

- **`examples/inference.ipynb`**：演示两类用法——(1) 从 checkpoint 创建
  `Policy` 并对一个假观测做推理；(2) 从 checkpoint 创建「活」的模型对象，
  用一个真实训练 batch 直接算 `compute_loss`——后者在不跑完整训练脚本的情况下，
  就能验证某个数据配置/模型配置是否能正确跑通前向。
- **`examples/policy_records.ipynb`**：配合 `PolicyRecorder` 产出的
  `step_*.npy` 文件，把 `inputs`/`outputs` 中的图像和关节角曲线可视化出来
  （用 `pandas`/`matplotlib`），用于逐步排查某次微调后策略在具体 step 上的行为。

### 8.7 工程化 / 开发者工具

- **训练回归测试**：`scripts/train_test.py` —— 强制 `JAX_PLATFORMS=cpu`，
  用 `debug` 配置跑 2 步训练、再以 `resume=True` 续跑 2 步，
  验证训练主循环和 checkpoint 恢复逻辑没有被破坏（`train_test.py:1-29`）。
- **配置系统的易用性保障**：注册表里配置名冲突会在 import 时直接报错，
  传错配置名时用 `difflib` 做模糊匹配给出"你是不是想输入 xxx"的提示（§3.1）。
- **代码风格与提交流程**：`pre-commit install` 之后，提交前跑
  `ruff check .` 与 `ruff format .`（`CONTRIBUTING.md:32`）；
  贡献流程（issue / feature request / PR 规范）见 `CONTRIBUTING.md`。
- **Troubleshooting 速查表**（README.md:309-323）：覆盖 `uv sync` 依赖冲突、
  训练 OOM（建议设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`、调 `fsdp_devices`
  或关 EMA）、policy server 连接失败、缺 norm stats、数据集下载失败、
  动作维度不匹配、训练 loss 发散（提示检查 `norm_stats.json` 里过小的
  `q01`/`q99`/`std`）等高频问题及对应解法。

---

## 9. 实操指南：没有实机，从微调到部署跑一遍完整链路

> 注意：仓库里没有"从零预训练"的入口——所有训练配置的 `weight_loader` 都是
> 从某个 `gs://openpi-assets/checkpoints/...` 基座 checkpoint 继续训练（§3.1、§8.1）。
> 所以"预训练到部署"在这个仓库里实际能做的是：**基于 base checkpoint 微调 → 存
> checkpoint → 起 policy server → 在仿真里闭环执行**。本章给出两条可以直接抄的
> 命令路径，分别对应"先确认链路没堵"和"跑一个有实际意义的微调"。两条路径都用
> **LIBERO**（MuJoCo 仿真 benchmark），因为它是 README 官方的微调教程范例，
> 还自带 Docker Compose 一体化评测和可核对的 benchmark 数字（§6、§8.5）。

### 9.1 这台机器的现状核对

写这份指南之前实际检查过这台机器（而不是假设），结论：

| 项目 | 现状 |
|---|---|
| GPU | 4 块 `NVIDIA L20`（46GB），其中 **GPU 0 已被其他任务占用约 34GB**，GPU 1/2/3 空闲 |
| Docker | 已安装（`29.1.3`） |
| uv | 已安装（`0.11.24`），但 `.venv` 还没建（没跑过 `uv sync`） |
| git submodule | `third_party/aloha`、`third_party/libero` 已经 `init` 过，不需要再 `git submodule update` |
| wandb | `~/.netrc` 里已有凭证，`train.py` 默认的 `wandb_enabled=True` 可以直接用，不用额外 `wandb login` |
| 磁盘 | `/local/mnt2` 剩 1.4TB，足够放 checkpoint 和数据集缓存 |
| 已下载的资产 | `~/.cache/openpi`（checkpoint 缓存）和本地 `assets/`（norm stats）都还是空的，`~/.cache/huggingface` 里也没有 libero 数据集——**两条路径的第一次运行都会触发下载**（base checkpoint 几 GB、`physical-intelligence/libero` 数据集若干 GB），视网络速度可能要等一会，下载完会被缓存，之后重跑不会再下 |

**单卡显存够不够**：对照 §8.2 的显存门槛表，LoRA 微调（>22.5GB）单张 L20（46GB）
轻松够用；全量微调（>70GB）单卡放不下，需要用 `--fsdp-devices` 把参数切到
多张卡上（这台机器空闲的 GPU 1/2/3 正好可以凑 3 张）。
下面所有单卡命令都显式用 `CUDA_VISIBLE_DEVICES=1`，避开被占用的 GPU 0。

### 9.2 两条路径共用的环境准备

```bash
cd /local/mnt2/workspace/danwa/project/VLA/openpi

# 装依赖（第一次跑需要几分钟）
uv sync
uv pip install -e .

# LIBERO 仿真的 docker 构建需要能访问 X11（本机如果是远程 ssh 没有图形界面，
# 可以跳过这一步，直接用"路径 A"里不需要渲染画面的验证方式）
sudo xhost +local:docker
```

### 9.3 路径 A：快速验证全链路（几分钟，几十步）

目标不是训出一个有用的策略，而是确认"微调 → checkpoint → serve → 推理"这条
链路在**真实数据、真实模型路径**上（不是 `debug` 配置的假数据）完全走得通，
中途不报错。用的是真正的 LoRA 配置 `pi0_libero_low_mem_finetune`
（`training/config.py:679-698`，π0 + LoRA，显存需求小，详见 §2.5、§3.5）。

```bash
# 1. 算 norm stats——用 --max-frames 只扫一小部分数据，而不是扫完整个数据集
#    （第一次跑会顺带把 physical-intelligence/libero 数据集下载到 HF 缓存）
CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py \
    --config-name=pi0_libero_low_mem_finetune --max-frames 2000

# 2. 跑 20 步训练，把默认的 30000 步 / 1000 步存档间隔 / 100 步打印间隔全部覆盖掉
#    （第一次跑会顺带把 pi0_base 基座 checkpoint 下载到 ~/.cache/openpi）
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=smoke_test --overwrite \
    --num-train-steps=20 --save-interval=10 --log-interval=1 --batch-size=8

# 3. 用第 20 步的 checkpoint 起一个 policy server
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_libero_low_mem_finetune \
    --policy.dir=checkpoints/pi0_libero_low_mem_finetune/smoke_test/20
```

另开一个终端，用合成观测测试服务端（§5.7），不需要 Mujoco/X11，几秒钟跑完：

```bash
uv run examples/simple_client/main.py --env LIBERO
```

看到连续打印的推理延迟统计、没有报错，就说明整条链路（微调→落盘→serve→
infer→拿到形状正确的 action）跑通了。如果不想要这次 smoke test 污染 wandb
面板，训练命令可以再加一个 `--no-wandb-enabled`。

### 9.4 路径 B：真实 LoRA 微调 + LIBERO 仿真闭环部署

这一条是真的训、真的在仿真里闭环跑，用来验证"微调出来的策略确实能让仿真机器人
把任务做对"，而不只是链路通不通。

```bash
# 1. 这次不限制帧数，扫一遍真实数据算出靠谱的 norm stats
CUDA_VISIBLE_DEVICES=1 uv run scripts/compute_norm_stats.py --config-name=pi0_libero_low_mem_finetune

# 2. 跑有意义的步数（默认配置是 30000 步；LoRA 收敛通常比全量微调快，
#    先跑几千步看看效果、不满意再续训也可以，--resume 可以接着训）
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=real_run --overwrite --num-train-steps=5000
```

训练过程中可以在 wandb 面板看 loss 曲线；checkpoint 每 1000 步存一次
（`save_interval`），每 5000 步的那个会被永久保留（`keep_period`，§3.7）。

跑完后用 Docker Compose 起"policy server + LIBERO 仿真评测"一体化流程
（§5.5、§6，`examples/libero/compose.yml`）：

```bash
export SERVER_ARGS="policy:checkpoint --policy.config pi0_libero_low_mem_finetune --policy.dir ./checkpoints/pi0_libero_low_mem_finetune/real_run/5000"
# 先只挑一个任务套件、少量 rollout，跑得快一些（默认是全部套件、每个任务 50 次 rollout）
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 5"

docker compose -f examples/libero/compose.yml up --build
# 如果报 egl 相关错误，改用：
# MUJOCO_GL=glx docker compose -f examples/libero/compose.yml up --build
```

评测脚本（`examples/libero/main.py:34-38`）会逐任务打印成功率。拿这个数字去对
README 里 π0.5 @ 30k 的 96.85%（`examples/libero/README.md`
"Results" 一节）：

> **注意这不是同一个设置的严格对照**——README 的 96.85% 是 `pi05_libero`
> （π0.5，**全量微调**，30000 步）跑出来的；上面命令用的是 `pi0_libero_low_mem_finetune`
> （π0，**LoRA**，可能只跑了 5000 步）。步数更少、是 LoRA 而不是全量微调，
> 成功率显著更低是预期之中的——这条路径的目的是验证"微调确实能学到东西、
> 部署链路确实能闭环工作"，不是复现论文数字。

如果真的想复现 96.85% 这个数字，要换成全量微调的 `pi05_libero`
（`training/config.py:744-763`），显存需求 >70GB，单卡放不下，
需要用 `--fsdp-devices` 切到多张卡（这台机器 GPU 1/2/3 空闲，正好 3 张）：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 uv run scripts/compute_norm_stats.py --config-name=pi05_libero
CUDA_VISIBLE_DEVICES=1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero \
    --exp-name=real_run_pi05 --overwrite --fsdp-devices=3
```

注意 `pi05_libero` 的 `batch_size=256`、`num_train_steps=30000`
是官方复现用的设置（`training/config.py:751,762`），完整跑完预计是数小时到
更长的量级（具体取决于这 3 张 L20 的实际吞吐，没有在这台机器上实测过，
不要直接当作承诺的时间）；中途可以用 `--resume` 接着训，也可以先用较小的
`--num-train-steps` 看中间效果。

### 9.5 两条路径的关键差异与常见坑

| | 路径 A（快速验证） | 路径 B（真实微调） |
|---|---|---|
| 目的 | 确认链路没有堵点 | 确认微调出的策略真的有效 |
| 配置 | `pi0_libero_low_mem_finetune`（LoRA） | 同上，或 `pi05_libero`（全量，对标 README 数字） |
| norm stats | `--max-frames 2000` 糊弄过去 | 不限制，扫全量数据 |
| 训练步数 | 20 步 | 数千步起，对标 README 需要 30000 步 |
| 验证方式 | `examples/simple_client`（合成观测，无需渲染） | `examples/libero` 的 Docker Compose 闭环仿真 |
| 单卡够不够 | 够（LoRA，L20 46GB 绰绰有余） | LoRA 够；全量微调 >70GB 需要 `--fsdp-devices` 多卡 |
| 耗时 | 几分钟 | 几十分钟到数小时，取决于步数 |

常见坑：
- **三个脚本传配置名的方式各不相同**，照抄另一个脚本的写法会直接报
  "unrecognized arguments"：

  | 脚本 | 配置名怎么传 | 底层 CLI 写法 |
  |---|---|---|
  | `train.py` | **位置参数**：`train.py pi0_libero_low_mem_finetune` | `_config.cli()` → `tyro.extras.overridable_config_cli`（`config.py:978-979`），每个配置注册成一个子命令 |
  | `compute_norm_stats.py` | **`--config-name=xxx`** | `tyro.cli(main)`，参数从 `main(config_name: str, ...)` 的函数签名推导，普通参数变成选项 |
  | `serve_policy.py` | **`--policy.config=xxx`** | `tyro.cli(Args)`，`Args` 是嵌套 dataclass；命令里的 `policy:checkpoint` 只是联合类型的子命令选择器，不是配置名 |

  （`data_loader.py:179` 那条 "Normalization stats not found" 的报错信息里写的也是
  `--config-name=<your-config>`，可以作为参照。）
- **第一次跑一定会下载东西**：`pi0_base`/`pi05_base` checkpoint（`gs://openpi-assets`）
  和 `physical-intelligence/libero` 数据集（HuggingFace Hub）都不在本地缓存里，
  两条路径的第一步都会触发下载，网络慢的话这是最耗时的部分，但只会发生一次（§4.5）。
- **GPU 0 已被占用**：所有示例命令都显式设了 `CUDA_VISIBLE_DEVICES`，
  避开已经在用的 GPU 0；真的要用全部 4 张卡之前，先用 `nvidia-smi` 确认一下它还占不占。
- **`--exp-name` 和 `--overwrite`/`--resume` 不能同时乱用**：同一个 `exp_name`
  重跑时，不加 `--overwrite` 也不加 `--resume` 会直接报错（`TrainConfig.__post_init__`
  校验 `resume` 和 `overwrite` 不能同时为真，`config.py:554-556`；两者都不给、
  目录又已存在时 Orbax 同样会拒绝覆盖）。
- **`pi05_libero` 没有 LoRA 版本**：§2.5 已经指出注册表里不存在任何
  `pi05_*_lora` 配置，想要全量复现 README 数字、又想省显存，只能靠
  `--fsdp-devices` 多卡分摊，而不是切到 LoRA。
- **LIBERO Docker 评测需要 X11**：纯 SSH 远程、没有 `$DISPLAY` 的机器上，
  `sudo xhost +local:docker` 和 `compose.yml` 里挂载的 `/tmp/.X11-unix`
  可能没法正常工作——这种情况下路径 A 的 `simple_client` 验证方式（不需要渲染）
  是更现实的选择。
