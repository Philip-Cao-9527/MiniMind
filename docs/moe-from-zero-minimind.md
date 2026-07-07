# MiniMind 里的 MoE：从 0 开始理解到源码落地

## 证据说明

- 本机已验证事实：本轮已在 `/home/harry/projects/MiniMind` 读取 [AGENTS.md](../AGENTS.md)、[README.md](../README.md)、[dataset/lm_dataset.py](../dataset/lm_dataset.py#L37)、[model/model_minimind.py](../model/model_minimind.py#L10)、[trainer/train_pretrain.py](../trainer/train_pretrain.py#L24)、[trainer/trainer_utils.py](../trainer/trainer_utils.py#L18)，并确认当前解释器为 `Python 3.12.3`。
- 当前本地代码事实：MiniMind 本地仓库已经存在 `MoE` 结构实现，关键落点在 [model/model_minimind.py](../model/model_minimind.py#L148)；预训练入口已经支持 `--use_moe`，见 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L100)。
- 上游引用事实：当前本地仓库还没有同步 `eval_llm.py`、`train_full_sft.py`、`train_lora.py` 等推理和更多训练入口，所以推理相关说明需要引用上游路径 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L12) 与 [上游 README.md](../../../references/minimind/README.md#L563)。
- 未验证边界：本轮没有实际运行 `use_moe=1` 的训练或推理命令，所以不能把“源码已支持 MoE”写成“本机已经跑通 MoE 训练 / 推理实验”。

## 可直接口述回答（>=1000字）

如果让我从 0 开始讲 MiniMind 里的 MoE，我会先给一个最短结论：MoE 的全称是 Mixture of Experts，中文通常叫“专家混合”或者“专家路由前馈层”。它不是把整个 Transformer 复制很多份，也不是让每个 token 经过所有分支再平均，而是把原来每层里“所有 token 都走同一个 MLP”的那一段，改成“先让一个路由器判断这个 token 更适合交给哪个 expert，再只激活少数 expert 去处理”。这样做的核心目的，是在不让每个 token 都付出全部计算代价的前提下，把模型总容量做大。

在 MiniMind 里，这件事落在每个 Decoder block 的前馈层位置。也就是说，Attention 还是那个 Attention，RoPE 还是那个 RoPE，RMSNorm 还是那个 RMSNorm，变化的重点不是注意力，而是 block 里的 FFN。你可以直接看 [model/model_minimind.py](../model/model_minimind.py#L178)：`MiniMindBlock` 在 [model/model_minimind.py](../model/model_minimind.py#L184) 用一句话决定是普通 `FeedForward` 还是 `MOEFeedForward`。如果 `config.use_moe=False`，走 dense MLP；如果 `config.use_moe=True`，就把这一层的 FFN 换成 MoE。

MoE 可以先用一个生活比喻理解。普通 dense MLP 像一家小诊所，不管你是牙疼、发烧还是扭伤，都会先见同一个全科医生；MoE 像一家分诊医院，门口先有一个分诊台，它不直接治病，只负责判断“这个病人更该去骨科、口腔科还是呼吸科”，然后把你分给少数几个更合适的专家。这里的“分诊台”就是 router，也就是代码里的 `self.gate`；“骨科、口腔科、呼吸科”就是代码里的多个 expert，也就是 [model/model_minimind.py](../model/model_minimind.py#L153) 这一串 `FeedForward` 子模块。

如果只看 MiniMind 当前本地代码，MoE 的默认配置是非常清楚的：在 [model/model_minimind.py](../model/model_minimind.py#L40) 到 [model/model_minimind.py](../model/model_minimind.py#L45)，可以看到 `num_experts=4`、`num_experts_per_tok=1`、`router_aux_loss_coef=5e-4`。这说明默认是 4 个专家、每个 token 只选 1 个专家，也就是 top-1 routing。用公式写就是：

$$
p(h) = \operatorname{softmax}(W_r h)
$$

这里 $h$ 是某个 token 的隐藏状态，$W_r$ 是路由器参数，$p(h)$ 是这个 token 对每个 expert 的打分概率。MiniMind 接着会选分数最高的 expert：

$$
e^* = \arg\max_e p_e(h)
$$

然后只让这个 expert 去处理这个 token。因为当前默认是 top-1，并且 [model/model_minimind.py](../model/model_minimind.py#L161) 开启了 `norm_topk_prob`，所以 top-1 时归一化后的权重其实就是 1，这意味着单个 token 的输出可以近似理解为：

$$
f_{\text{MoE}}(h) = f_{e^*}(h)
$$

这里的 $f_{e^*}$ 就是被选中的那个 expert 对应的前馈网络。

为什么 MiniMind 要这么做？因为普通 dense MLP 的问题是：每个 token 无论内容是什么，都走同一套大矩阵。这样简单、稳定，但容量和计算基本绑死。MoE 的思路是，把“一个很大的通用前馈层”拆成“多个更有分工的专家前馈层”，每个 token 只激活少数几个 expert，于是模型总参数量可以更大，但每次真正参与计算的活跃参数不一定同步线性增长。这个思路在 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L18) 到 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L28) 也有体现，那里会区分总参数和 active 参数，并打印成 `198M-A64M` 这种形式。它想表达的是：模型账面总容量很大，但单次 token 真正走过的那部分参数没有把所有 expert 都算进去。

MoE 在 MiniMind 训练链路里的位置，也不能脱离整个 LLM 流程单独理解。预训练数据先在 [dataset/lm_dataset.py](../dataset/lm_dataset.py#L37) 里被 tokenizer 编成 token id，再补 `bos/eos`，padding 后得到 `input_ids` 和 `labels`。然后 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L36) 会执行 `res = model(input_ids, labels=labels)`。这个 `model` 会一路进入 [model/model_minimind.py](../model/model_minimind.py#L245) 的 `MiniMindForCausalLM.forward`。如果当前 block 用的是 MoE，那么每层 FFN 都会先路由再选 expert，最后所有层的辅助损失在 [model/model_minimind.py](../model/model_minimind.py#L231) 累加成 `aux_loss`。主语言模型损失仍然是 shifted cross entropy，在 [model/model_minimind.py](../model/model_minimind.py#L251) 到 [model/model_minimind.py](../model/model_minimind.py#L252) 计算；训练总损失则在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L37) 变成：

$$
L_{\text{train}} = L_{\text{CE}} + L_{\text{aux}}
$$

其中 $L_{\text{CE}}$ 是“预测下一个 token 对不对”，$L_{\text{aux}}$ 是“路由有没有过度偏向少数 expert”。前者负责学语言，后者负责让专家不要全部挤到一个桶里。再往后，MiniMind 会在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L40) 做 `backward()`，在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L46) 用 optimizer 更新参数，在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L61) 之后保存模型权重，并在 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L63) 到 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L67) 里给 MoE 权重自动加 `_moe` 后缀。

推理阶段也能看到 MoE 的痕迹。虽然当前本地仓库还没有同步 `eval_llm.py`，但上游入口 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L12) 到 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L23) 会根据 `--use_moe` 决定是否构造 `MiniMindConfig(..., use_moe=True)`，并从 `xxx_moe.pth` 这类权重文件加载。也就是说，MoE 对推理来说不是“额外再套一层逻辑”，而是模型结构本身的一部分。只不过在推理模式下，辅助损失不再参与训练目标，MoE 只负责“这个 token 该交给哪个 expert”。

最后一定要强调边界。根据本轮能确认的事实，我可以说“MiniMind 当前本地源码已经实现了 MoE FFN、默认 4 experts / top-1 routing、训练入口支持 `--use_moe`、checkpoint 命名会加 `_moe` 后缀、上游推理入口支持加载 MoE 权重”。但我不能说“我已经在这台机器上跑通了 MoE 训练并验证吞吐”，因为本轮并没有执行那个实验。对于面试或复盘，最稳妥的表述是：我已经基于 MiniMind 当前本地源码读通了 MoE 在 `tokenizer -> dataset -> model -> loss -> backward -> optimizer -> checkpoint -> inference` 这条链路中的落点，但本机 MoE 训练和推理结果仍需后续 smoke test 进一步验证。

## 详细原理讲解（>=3000字，含公式）

### 1. 先别急着想“专家”，先想普通 MLP 在干什么

很多人一看到 MoE，会以为它是一个“比 Transformer 更高级的新模型”。其实在 MiniMind 里，MoE 没有推翻 Transformer 主干，它只是把每层 block 中原本的前馈网络换成了“带路由的前馈网络”。这点非常重要，因为如果你连普通 FFN 在做什么都没弄清楚，就会把 MoE 理解成一种很玄的结构。

在 Decoder-only Transformer 里，一个 block 大致可以拆成两大块：一块是 self-attention，用来决定“当前位置该看前面哪些 token”；另一块是 FFN，也就是前馈网络，用来对每个位置的 hidden state 做非线性变换。MiniMind 普通 dense FFN 的实现就在 [model/model_minimind.py](../model/model_minimind.py#L136) 到 [model/model_minimind.py](../model/model_minimind.py#L146)。它不是单纯两层线性层，而是一个带门控的 SwiGLU 风格结构，可以写成：

$$
f_{\text{dense}}(h) =
W_{\text{down}}
\left(
\phi(W_{\text{gate}}h) \odot (W_{\text{up}}h)
\right)
$$

这里：

- $h \in \mathbb{R}^{H}$ 是某个 token 的隐藏状态。
- $W_{\text{gate}} \in \mathbb{R}^{I \times H}$ 是门控投影。
- $W_{\text{up}} \in \mathbb{R}^{I \times H}$ 是升维投影。
- $W_{\text{down}} \in \mathbb{R}^{H \times I}$ 是降维投影。
- $\phi(\cdot)$ 是激活函数，MiniMind 默认是 `silu`。
- $\odot$ 表示逐元素相乘。
- $H$ 是 hidden size，$I$ 是 intermediate size。

直觉上，它的作用像“先把一个 token 的内部表示扩展开，再做非线性筛选，最后再压回 hidden size”。dense FFN 的优点是简单、稳定、好并行；缺点是每个 token 都走完全一样的一套大矩阵，无论这个 token 是在写代码、讲数学、还是闲聊日常，它都不会动态换一套更擅长的参数。

这里可以用第一个比喻。dense FFN 很像“全班所有作业都交给同一个老师批改”。这个老师很全能，但数学题、作文、翻译题都让他一个人批，效率和专业分工都有限。MoE 的想法则是：不如让系统先判断这道题更像数学、语文还是翻译，然后交给更合适的老师。注意，这个比喻只用来帮助建立直觉，真正的实现仍然是张量、矩阵和路由索引，不是人工规则。

### 2. MoE 到底是什么：一个路由器加若干个专家

MoE 全称 Mixture of Experts，可以先把它拆成两个部件看：

1. 路由器 `router/gate`：负责给当前 token 对每个 expert 打分。
2. 专家 `experts`：每个 expert 本质上仍然是一个前馈网络。

在 MiniMind 里，这个结构落在 [model/model_minimind.py](../model/model_minimind.py#L148) 到 [model/model_minimind.py](../model/model_minimind.py#L176)。`self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)` 表示：对每个 token 的 hidden state 做一个线性映射，直接得到对所有 expert 的分数。然后用 `softmax` 转成概率：

$$
p(h) = \operatorname{softmax}(W_r h)
$$

这里：

- $W_r \in \mathbb{R}^{E \times H}$ 是路由器权重。
- $E$ 是 expert 数量。
- $p(h) \in \mathbb{R}^{E}$ 表示这个 token 对每个 expert 的偏好概率。

接下来不是所有 expert 都算一遍，而是只保留 top-$K$：

$$
\mathcal{K}(h) = \operatorname{TopK}(p(h), K)
$$

MiniMind 当前默认 `num_experts_per_tok=1`，也就是 $K=1$。这时每个 token 只保留分数最高的 1 个 expert。若设被选中的 expert 为 $e^*$，那么：

$$
e^* = \arg\max_e p_e(h)
$$

如果是更一般的 top-$K$ 路由，MoE 输出可以写成：

$$
f_{\text{MoE}}(h) =
\sum_{e \in \mathcal{K}(h)}
\tilde{p}_e(h)\, f_e(h)
$$

其中 $f_e(h)$ 是第 $e$ 个 expert 对这个 token 的输出，$\tilde{p}_e(h)$ 是归一化后的 top-$K$ 权重：

$$
\tilde{p}_e(h) =
\frac{p_e(h)}{\sum_{j \in \mathcal{K}(h)} p_j(h)}
$$

这正对应 [model/model_minimind.py](../model/model_minimind.py#L160) 到 [model/model_minimind.py](../model/model_minimind.py#L161) 的 `topk_weight` 和 `norm_topk_prob`。但因为 MiniMind 默认是 top-1，所以当前最容易记住的结论其实是：**默认情况下，一个 token 最终只走 1 个 expert，且这个 expert 的权重就是 1。**

这时公式直接简化为：

$$
f_{\text{MoE}}(h) = f_{e^*}(h)
$$

你可以把它理解成“每个 token 在这一层 FFN 前先选科室，再去对应科室问诊”。注意，这不等于“整个句子只走一个 expert”。因为路由是逐 token 做的，同一句话里的不同 token 可能被分给不同 expert。

### 3. 用一个极小数字例子，把 top-1 路由走一遍

下面我们不用大模型参数，只用极小数字把过程走一遍。

假设：

- batch size $B=2$
- seq length $T=3$
- hidden size $H=4$
- expert 数量 $E=3$
- 每个 token 选 1 个 expert，也就是 $K=1$

那么输入到某一层 MoE FFN 前的 hidden states 张量形状是：

$$
x \in \mathbb{R}^{B \times T \times H} = \mathbb{R}^{2 \times 3 \times 4}
$$

MiniMind 在 [model/model_minimind.py](../model/model_minimind.py#L157) 到 [model/model_minimind.py](../model/model_minimind.py#L159) 里先做：

$$
x_{\text{flat}} \in \mathbb{R}^{(B \cdot T) \times H} = \mathbb{R}^{6 \times 4}
$$

也就是把 6 个 token 摊平，每个 token 都独立接受路由器打分。假设这 6 个 token 的路由分数做 `softmax` 后分别是：

$$
\begin{aligned}
t_1 &: [0.10,\ 0.70,\ 0.20] \\
t_2 &: [0.60,\ 0.15,\ 0.25] \\
t_3 &: [0.05,\ 0.15,\ 0.80] \\
t_4 &: [0.55,\ 0.35,\ 0.10] \\
t_5 &: [0.20,\ 0.50,\ 0.30] \\
t_6 &: [0.34,\ 0.33,\ 0.33]
\end{aligned}
$$

那么 top-1 选择结果就是：

$$
\begin{aligned}
t_1 &\rightarrow expert\ 2 \\
t_2 &\rightarrow expert\ 1 \\
t_3 &\rightarrow expert\ 3 \\
t_4 &\rightarrow expert\ 1 \\
t_5 &\rightarrow expert\ 2 \\
t_6 &\rightarrow expert\ 1
\end{aligned}
$$

于是：

- expert 1 处理 token $t_2,t_4,t_6$
- expert 2 处理 token $t_1,t_5$
- expert 3 处理 token $t_3$

这正是 [model/model_minimind.py](../model/model_minimind.py#L163) 到 [model/model_minimind.py](../model/model_minimind.py#L168) 那个 `for i, expert in enumerate(self.experts)` 循环在干的事情。代码并不是让所有 token 都跑所有 expert，而是先根据 `topk_idx` 找出“当前这个 expert 真正该处理哪些 token”，然后只对这些 token 做前向，再用 `index_add_` 把结果写回总输出张量 `y`。

如果当前是 top-1，那么每个 token 只会被一个 expert 写回一次；如果是 top-2 或 top-4，则同一个 token 会被多个 expert 分别处理，最后按权重相加。

这里还有第二个比喻。你可以把 `x_flat` 想成一个快递分拣中心里的 6 个包裹。router 先给每个包裹贴“最适合哪个传送带”的标签，然后 expert 1、expert 2、expert 3 各自只处理属于自己的包裹。最后所有包裹还是会被放回原来的总清单位置。这个比喻对应的技术动作就是“按 expert 分桶，再写回原 token 位置”。

### 4. 再看一个 top-2 小例子，理解“专家混合”四个字为什么成立

虽然 MiniMind 默认是 top-1，但配置里 [model/model_minimind.py](../model/model_minimind.py#L42) 允许 `num_experts_per_tok` 改成更大的值。所以有必要顺带理解 top-2。

假设某个 token 的 router 概率是：

$$
p(h) = [0.50,\ 0.30,\ 0.20]
$$

如果取 top-2，那么被选中的 expert 是 1 和 2。归一化后的权重是：

$$
\tilde{p}(h) =
\left[
\frac{0.50}{0.50 + 0.30},
\frac{0.30}{0.50 + 0.30}
\right]
= [0.625,\ 0.375]
$$

假设两个 expert 的输出分别是：

$$
f_1(h) = [2,\ 0,\ 1,\ 3], \quad
f_2(h) = [0,\ 4,\ 1,\ 1]
$$

那么最终输出不是二选一，而是加权和：

$$
f_{\text{MoE}}(h)
= 0.625 \cdot f_1(h) + 0.375 \cdot f_2(h)
$$

也就是：

$$
f_{\text{MoE}}(h)
= [1.25,\ 0,\ 0.625,\ 1.875] + [0,\ 1.5,\ 0.375,\ 0.375]
= [1.25,\ 1.5,\ 1.0,\ 2.25]
$$

这时“mixture”这个词就特别直观了，因为输出确实是多个专家结果的混合。但再次强调：**MiniMind 当前默认不是这个模式，而是更省事、更容易讲清楚的 top-1。**

### 5. 对照 MiniMind 源码：MoE 不是全模型替换，而是 FFN 局部替换

理解 MoE 最容易犯的一个错，是以为“启用 MoE 后，整个 Transformer 都变成另一套结构了”。MiniMind 的源码恰好能帮你把这个误解掐掉。

先看配置。MoE 相关超参数在 [model/model_minimind.py](../model/model_minimind.py#L40) 到 [model/model_minimind.py](../model/model_minimind.py#L45)：

- `num_experts=4`：默认 4 个 expert。
- `num_experts_per_tok=1`：默认每个 token 只选 1 个 expert。
- `moe_intermediate_size`：expert 内部 MLP 的中间维度。
- `norm_topk_prob=True`：对 top-k 权重做归一化。
- `router_aux_loss_coef=5e-4`：路由辅助损失系数。

再看 block 级别切换。真正的“dense 还是 MoE”分叉发生在 [model/model_minimind.py](../model/model_minimind.py#L184)：

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

这句代码非常值钱，因为它告诉你：

1. Attention 路径没有换。
2. 残差连接没有换。
3. RMSNorm 没有换。
4. 只有 `mlp` 这个子模块被替换。

所以，如果你在讲 MiniMind 的 MoE，最准确的话不是“MiniMind 是 MoE Transformer”，而是“MiniMind 在 Decoder block 的 FFN 位置支持用 MoE 替换 dense MLP”。这样说更严谨，也更贴近代码事实。

然后看层级汇总。每层 MoE 的辅助损失不是各算各的就结束了，而是会在 [model/model_minimind.py](../model/model_minimind.py#L231) 被累加：

$$
L_{\text{aux}} = \sum_{\ell=1}^{L} L_{\text{aux}}^{(\ell)}
$$

这里 $L$ 是层数，当前默认是 8 层。最终 `MiniMindForCausalLM.forward` 会把 `loss`、`aux_loss`、`logits`、`past_key_values` 一起返回，见 [model/model_minimind.py](../model/model_minimind.py#L245) 到 [model/model_minimind.py](../model/model_minimind.py#L253)。

### 6. 把公式翻译成张量 shape，你就真的理解了

很多人对 MoE 的“概念”能讲两句，但一到 shape 就虚了。真正理解一个实现，最好能把每一步的张量形状说清楚。

在 MiniMind 的 `MOEFeedForward.forward(x)` 中，输入是：

$$
x \in \mathbb{R}^{B \times T \times H}
$$

其中：

- $B$ 是 batch size
- $T$ 是序列长度
- $H$ 是 hidden size

代码先做：

$$
x_{\text{flat}} = \operatorname{reshape}(x) \in \mathbb{R}^{N \times H}
$$

其中：

$$
N = B \cdot T
$$

然后 router 线性层输出：

$$
\text{logits}_{\text{router}} \in \mathbb{R}^{N \times E}
$$

softmax 后得到：

$$
\text{scores} \in \mathbb{R}^{N \times E}
$$

`topk` 后得到两个张量：

$$
\text{topk\_weight} \in \mathbb{R}^{N \times K}, \quad
\text{topk\_idx} \in \mathbb{R}^{N \times K}
$$

MiniMind 当前默认 $K=1$，所以这两个张量其实可以想成 $N \times 1$。随后进入按 expert 遍历的循环。对第 $i$ 个 expert，会先找到“哪些 token 的 top-k 里包含 expert $i$”。这些 token 的数量记作 $n_i$，那么进入该 expert 的张量形状就是：

$$
x_i \in \mathbb{R}^{n_i \times H}
$$

expert 输出仍然是：

$$
f_i(x_i) \in \mathbb{R}^{n_i \times H}
$$

如果是 top-1，$n_i$ 就是被分配给 expert $i$ 的 token 数。所有 expert 处理完后，代码把结果加回：

$$
y \in \mathbb{R}^{N \times H}
$$

最后 reshape 回：

$$
y_{\text{out}} \in \mathbb{R}^{B \times T \times H}
$$

所以从 block 外面看，MoE FFN 和 dense FFN 的输入输出 shape 完全一致，都是“输入一个 `[B, T, H]`，输出一个 `[B, T, H]`”。这就是为什么它能被无缝塞进现有 Transformer block，而不用改残差连接接口。

### 7. 为什么还要 aux loss：因为路由器很容易偏科

如果只做 top-1 路由而不加约束，router 很可能学成一个坏习惯：长期把大多数 token 都丢给同一个 expert。这样会带来两个问题：

1. 某些 expert 过载，另一些 expert 几乎不被训练。
2. 账面上你有很多专家，实际上只有少数人在工作。

所以 MiniMind 在训练时额外加了一个辅助损失，代码在 [model/model_minimind.py](../model/model_minimind.py#L171) 到 [model/model_minimind.py](../model/model_minimind.py#L173)。在当前默认 top-1 情况下，这个公式最容易解释成：

$$
load_e = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}[e_n = e]
$$

这里：

- $N=B \cdot T$，表示这一层当前总 token 数。
- $e_n$ 表示第 $n$ 个 token 被分到的 expert 编号。
- $\mathbf{1}[\cdot]$ 是指示函数，成立为 1，否则为 0。

`load_e` 表示 expert $e$ 实际接到的 token 比例。与此同时，router 对 expert 的平均偏好还可以写成：

$$
prob_e = \frac{1}{N} \sum_{n=1}^{N} p_e^{(n)}
$$

于是单层辅助损失可近似理解为：

$$
L_{\text{aux, layer}} =
\lambda \cdot E \cdot \sum_{e=1}^{E} load_e \cdot prob_e
$$

这里：

- $\lambda$ 对应 `router_aux_loss_coef`
- $E$ 是 expert 数

注意我这里用了“近似理解”这四个字，是因为当前源码的 `load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)` 在一般 top-$K$ 情况下会保留一个 top-k 槽位维度，然后借助广播与 `scores.mean(0)` 相乘；但在 MiniMind 默认 top-1 条件下，它就可以非常直观地理解成“每个 expert 被选中的比例”。这也是本轮文档为什么优先围绕 top-1 讲。

我们再用一个很小的数字例子。假设某层一共只有 4 个 token，3 个 expert，top-1 路由结果如下：

$$
t_1 \rightarrow expert\ 1,\quad
t_2 \rightarrow expert\ 1,\quad
t_3 \rightarrow expert\ 1,\quad
t_4 \rightarrow expert\ 2
$$

那么：

$$
load = [0.75,\ 0.25,\ 0.00]
$$

如果 4 个 token 对 3 个 expert 的平均路由概率是：

$$
prob = [0.50,\ 0.30,\ 0.20]
$$

并设 $\lambda = 5 \times 10^{-4}$，$E=3$，则：

$$
L_{\text{aux, layer}}
= 5 \times 10^{-4} \times 3 \times
(0.75 \times 0.50 + 0.25 \times 0.30 + 0.00 \times 0.20)
$$

$$
= 5 \times 10^{-4} \times 3 \times 0.45
= 6.75 \times 10^{-4}
$$

这个值本身不重要，重要的是它在提醒 router：你别老把 token 往 expert 1 那里塞。辅助损失不是直接提高回答质量的魔法，而是帮助训练出“更多专家都能学到东西”的路由行为。

### 8. 从 tokenizer 到 checkpoint：MoE 在 MiniMind 全链路里到底出现在哪里

MoE 不能只停留在“模型层里有个 router”。如果你真要讲明白，必须把它放回完整训练闭环。

#### 8.1 tokenizer 和 dataset 阶段

预训练数据读取在 [dataset/lm_dataset.py](../dataset/lm_dataset.py#L37)。`PretrainDataset.__getitem__` 会：

1. 从 `sample['text']` 取文本。
2. 用 tokenizer 编码。
3. 手动加上 `bos` 和 `eos`。
4. padding 到固定长度。
5. 复制出 `labels`，并把 pad 位置置成 `-100`。

所以这一阶段还没有 MoE。它只负责准备训练样本。这里最核心的公式是 shifted language modeling 的监督关系：

$$
x = \text{logits}[:, :-1, :]
$$

$$
y = \text{labels}[:, 1:]
$$

也就是说当前位置预测的是下一个 token，而不是当前位置自己。

#### 8.2 model forward 阶段

进入模型后，先是 embedding、RoPE、Attention、残差，然后来到每一层 block 的 FFN 位置。如果 `use_moe=False`，这一步是普通 `FeedForward`；如果 `use_moe=True`，这一层就切成“router + experts”的路由前馈层。

关键是：MoE 只影响这一层 FFN 的内部计算方式，不影响 `input_ids` 的语义，不改变 `labels` 的构造，也不改变 Causal LM 最后的 `lm_head` 预测方式。

#### 8.3 loss 阶段

语言模型主损失在 [model/model_minimind.py](../model/model_minimind.py#L251) 到 [model/model_minimind.py](../model/model_minimind.py#L252)：

$$
L_{\text{CE}}
= - \frac{1}{|S|}
\sum_{(b,t)\in S}
\log
\frac{e^{z_{b,t,y_{b,t+1}}}}
{\sum_{v=1}^{V} e^{z_{b,t,v}}}
$$

其中：

- $z_{b,t,v}$ 是第 $b$ 条样本、第 $t$ 个位置、词表第 $v$ 个 token 的 logit。
- $y_{b,t+1}$ 是目标 token id。
- $V$ 是词表大小。
- $S$ 是所有 `labels[b, t+1] != -100` 的有效监督位置。

而训练入口在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L37) 把总损失写成：

$$
L_{\text{train}} = L_{\text{CE}} + L_{\text{aux}}
$$

如果用了梯度累积，真正送入 `backward()` 的是：

$$
L_{\text{backward}} =
\frac{L_{\text{train}}}{\text{accumulation\_steps}}
$$

所以 MoE 并不是替代交叉熵，而是在交叉熵旁边再加一个“路由均衡项”。

#### 8.4 backward 和 optimizer 阶段

[trainer/train_pretrain.py](../trainer/train_pretrain.py#L40) 的 `scaler.scale(loss).backward()` 会把梯度传回去。这里要特别注意，MoE 的梯度有两部分来源：

1. 语言模型主损失对被激活 expert 参数的梯度。
2. 辅助损失对 router 权重的梯度。

在 top-1 路由下，被选中的 expert 会更直接地收到主任务梯度；未被选中的 expert 如果长期选不到，就很难学到东西，所以才需要 aux loss 帮忙维持更健康的负载分布。

接着 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L43) 到 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L49) 会做：

1. `unscale_`
2. `clip_grad_norm_`
3. `step`
4. `update`
5. `zero_grad`

真正更新参数的是 optimizer step，而不是 backward 本身。

#### 8.5 checkpoint 阶段

MoE 对 checkpoint 最大的直接影响之一，是权重命名和模型结构匹配。训练保存时会根据 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L63) 的 `moe_suffix = '_moe' if lm_config.use_moe else ''` 来决定文件名。`lm_checkpoint` 在 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L63) 到 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L67) 也会用同样规则。

这意味着：

- dense 模型权重和 MoE 模型权重文件名不同。
- 加载时如果结构不匹配，`state_dict` 也会对不上。
- 你不能拿 dense 的权重直接当成 MoE 权重用，反过来也不行，除非有额外转换逻辑。

### 9. 推理和 generate 阶段：MoE 还在，但职责变了

当前本地仓库没有同步推理入口文件，所以这一段只能引用上游 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L12)。但它足够说明链路。

上游 `init_model` 会根据 `--use_moe` 决定：

$$
\text{config.use\_moe} \in \{False, True\}
$$

如果启用 MoE，就从带 `_moe` 后缀的权重加载。之后 `generate` 会循环执行：

$$
\text{input ids}
\rightarrow \text{forward}
\rightarrow \text{logits}_{t}
\rightarrow \text{sampling}
\rightarrow \text{next token}
$$

在 [model/model_minimind.py](../model/model_minimind.py#L257) 到 [model/model_minimind.py](../model/model_minimind.py#L288)，`generate` 还会配合 `past_key_values` 做 KV Cache，只把新增 token 送入后续 forward。这个流程和 dense 模型一致，区别只是“每次经过 block 的 FFN 时，是普通 MLP 还是 MoE FFN”。

还有一个很容易忽略的点：推理时模型一般处于 `eval()` 模式，所以 MoE 的 `aux_loss` 不再承担训练约束。你看 [model/model_minimind.py](../model/model_minimind.py#L171) 这一段就能发现，辅助损失只在 `self.training` 且 `router_aux_loss_coef > 0` 时才会被真正计算；否则返回零张量。所以推理里保留的是“路由选择能力”，不是“再继续做负载均衡训练”。

### 10. 常见误区和高频易错点

#### 误区一：MoE 等于把整个模型复制了很多份

不对。MiniMind 当前实现里，变化的是 FFN 子模块，不是整个 Transformer 主干都复制。Attention、Embedding、RoPE、Norm、`lm_head` 都还在主干里共享。

#### 误区二：MoE 参数更多，所以训练一定更快

不对。参数更多和训练更快不是一回事。上游说明 [上游 README.md](../../../references/minimind/README.md#L563) 到 [上游 README.md](../../../references/minimind/README.md#L578) 明确写到，当前原生 PyTorch 实现下，expert 变多后，token 分桶、kernel 启停和调度开销会明显增加；`4 experts / top-1` 是一个现实折中。这个说法是上游文档事实，不是我在本机实测出来的结论。

#### 误区三：既然默认 top-1，那其他 expert 不就没用了

也不对。top-1 只表示“每个 token 在一层里只激活 1 个 expert”，不表示“整个 batch 只用 1 个 expert”。不同 token 完全可能被分配到不同 expert，所以多个 expert 仍然会共同工作。

#### 误区四：aux loss 越大越好

不对。aux loss 不是主任务指标，它只是辅助路由更均衡。你不能把它当成“模型能力提升”的直接证据。真正回答质量仍然主要取决于主语言模型损失、数据质量、训练规模和推理设置。

#### 误区五：MoE 会改变 labels 语义

不会。`labels` 仍然是语言模型的下一个 token 监督。MoE 改的是 hidden states 在 FFN 内部如何变换，不改 dataset 的监督定义。

### 11. 在 RTX 5060 Laptop 约 8GB 显存上的现实边界

这一段必须谨慎说，因为本轮没有实测 MoE 训练。

可以肯定的部分有两类：

1. 项目规则里已经明确本机现实边界大约是 RTX 5060 Laptop 8GB 显存，见 [AGENTS.md](../AGENTS.md) 的训练与显存约束。
2. 上游文档说明 MoE 虽然 active 参数更省，但原生 PyTorch 训练会多出明显的分桶和调度开销，见 [上游 README.md](../../../references/minimind/README.md#L565) 到 [上游 README.md](../../../references/minimind/README.md#L578)。

因此在没有跑实验前，最稳妥的工程判断是：

- 8GB 显存下不要直接照搬上游完整训练参数。
- 更适合先做极小 batch、短序列、少步数的 smoke test。
- 即使 active 参数看起来不大，总参数、优化器状态、激活、缓存和分桶开销也可能让显存或速度吃紧。
- 原生 PyTorch 版 MoE 更偏“源码理解和最小验证友好”，不代表它是 8GB 单卡下的最佳性能方案。

更保守一点说，当前你能写进实验计划的是“拟做 MoE smoke test”，不能写“MoE 已在本机 8GB 显存稳定训练”。

### 12. 一个最小验证思路：以后如果要真验证，应该怎么做

虽然本轮没有实际运行，但从代码逻辑出发，可以设计一个很小的验证闭环：

1. 设 `hidden_size` 和 `num_hidden_layers` 用默认值或更小值。
2. 打开 `--use_moe 1`。
3. 用极小 `max_seq_len` 和极小 batch。
4. 跑 1 个 batch。
5. 记录 `loss`、`logits_loss`、`aux_loss` 是否都是有限值。
6. 确认是否能生成 `*_moe.pth` 和 `*_moe_resume.pth`。

如果做到了这些，你才能进一步说“本机至少完成了 MoE 单 batch 训练闭环验证”。再往后，才轮到更长训练、吞吐、显存、收敛和推理质量问题。

一个很实用的记忆点是：

$$
\text{MoE} =
\text{dense FFN 的局部替换}
+ \text{逐 token 路由}
+ \text{少数 expert 激活}
+ \text{aux loss 维持负载均衡}
$$

如果你能把上面这四件事和 MiniMind 代码位置一一对上，MoE 基本就真正入门了。

## 项目落地点

### 已实现

- 当前本地仓库已经实现 MoE 结构配置，见 [model/model_minimind.py](../model/model_minimind.py#L40)。
- 当前本地仓库已经实现 `MOEFeedForward`，见 [model/model_minimind.py](../model/model_minimind.py#L148)。
- 当前本地仓库已经在 block 级别支持 dense / MoE 切换，见 [model/model_minimind.py](../model/model_minimind.py#L184)。
- 当前本地仓库已经在预训练入口支持 `--use_moe`，见 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L100)。
- 当前本地仓库已经在 checkpoint 命名中区分 `_moe` 后缀，见 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L63)。

### 上游引用路径

- 当前本地仓库尚未同步推理入口；推理加载 MoE 权重的说明来自 [上游 eval_llm.py](../../../references/minimind/eval_llm.py#L12)。
- 关于 `4 experts / top-1` 和训练开销折中的说明，来自 [上游 README.md](../../../references/minimind/README.md#L563)。

### 后续可扩展

- 可以补一个 `tests/test_moe_forward.py`，验证 `use_moe=True` 时 `forward` 的输出 shape、`aux_loss` 是否为有限值。
- 可以补一个极小 `MoE` 单 batch smoke test，确认 `loss -> backward -> optimizer.step -> 保存 checkpoint` 闭环。
- 可以补一份 `docs/experiment-moe-smoke-test-*.md`，专门记录 8GB 显存下的参数组合与结果。

### 需要验证

- 本机是否能在 `RTX 5060 Laptop 8GB` 上稳定跑通 `use_moe=1` 的单 batch 训练。
- 本机 `*_moe.pth` 权重是否能成功被上游推理入口加载并生成文本。
- `aux_loss` 的数值范围、训练初期的波动情况、以及它和 `logits_loss` 的相对量级。

### 哪些现在能写进 README、实验记录或面试材料，哪些不能写

- 现在可以写：本地源码已经实现并读通 MiniMind 的 MoE FFN、top-1 routing、aux loss 与训练入口开关。
- 现在可以写：已经基于源码梳理出 MoE 在 `dataset -> model -> loss -> backward -> checkpoint -> inference` 链路中的落点。
- 现在不能写：本机已经完成 MoE 训练、MoE 推理、MoE 吞吐对比、MoE 收敛效果或 MoE 输出质量评测。
- 更保守的替代表述：把“完成 MoE 训练验证”改成“完成 MoE 源码阅读与实验方案设计，训练 / 推理结果待后续 smoke test 验证”。

## 面试官 / 评审者可能追问与回答

### 追问 1：MoE 为什么能做到“总参数更大，但活跃参数不一定同样大”？

答：因为不是每个 token 都经过所有 expert。MiniMind 在 [model/model_minimind.py](../model/model_minimind.py#L159) 到 [model/model_minimind.py](../model/model_minimind.py#L168) 里先做 router 打分，再只让 top-k expert 前向。当前默认是 top-1，所以一个 token 在某层只激活 1 个 expert。也正因为如此，[trainer/trainer_utils.py](../trainer/trainer_utils.py#L18) 到 [trainer/trainer_utils.py](../trainer/trainer_utils.py#L28) 才会区分总参数和 active 参数。这里我能确认的是“源码这样设计了”，但还没有做本机显存和吞吐实测。

### 追问 2：MiniMind 的 MoE 和 dense 模型相比，究竟改了哪些地方？

答：本地代码里最核心的变化只有 FFN 子模块。`MiniMindBlock` 在 [model/model_minimind.py](../model/model_minimind.py#L184) 根据 `use_moe` 选择 `FeedForward` 或 `MOEFeedForward`。Attention、RoPE、RMSNorm、残差连接、`lm_head` 都还在。更准确的话应该说“MiniMind 支持用 MoE 替换 block 内的 dense FFN”，而不是泛泛地说“整个模型都换成了另一套架构”。

### 追问 3：如果去掉 aux loss，会发生什么？

答：最直接的风险是 router 长期偏向少数 expert，导致有些 expert 很少被激活，训练不充分。MiniMind 在 [model/model_minimind.py](../model/model_minimind.py#L171) 到 [model/model_minimind.py](../model/model_minimind.py#L173) 显式计算了 `aux_loss`，训练入口又在 [trainer/train_pretrain.py](../trainer/train_pretrain.py#L37) 把它加到总损失里。我目前能确认它的计算方式和代码路径，但没有做“去掉 aux loss 前后收敛对比”的本机实验，所以不能宣称具体效果差异。

### 追问 4：MiniMind 默认 top-1 routing，有什么含义？

答：它表示每个 token 在每层 MoE FFN 中只选择 1 个 expert。对应本地配置是 [model/model_minimind.py](../model/model_minimind.py#L42) 的 `num_experts_per_tok=1`。这会让解释更直观，也能减少每个 token 的激活计算量。由于 [model/model_minimind.py](../model/model_minimind.py#L161) 默认会归一化 top-k 权重，所以 top-1 时这个权重其实就是 1，也就是某个 token 的输出基本等于被选中 expert 的输出。

### 追问 5：既然上游说 `4 experts / top-1` 只比 dense 慢约 50%，你能直接把这当成自己的实验结论吗？

答：不能。那是上游文档 [上游 README.md](../../../references/minimind/README.md#L565) 到 [上游 README.md](../../../references/minimind/README.md#L566) 的说明，不是我本轮在本机复现得到的数字。更保守、也更诚实的表达应该是：上游给出了 native PyTorch 实现下的经验描述，而我当前只完成了本地源码阅读和链路拆解，尚未做本机吞吐与显存对比实验。
