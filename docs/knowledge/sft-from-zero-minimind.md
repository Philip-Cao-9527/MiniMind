# 从 0 理解 MiniMind 里的 SFT

## 事实边界说明

- 本机已验证事实：
  本轮已实际读取 [AGENTS.md](../AGENTS.md)、[README.md](../README.md)、[源码导览](./minimind-source-guide.md)、[dataset/lm_dataset.py](../dataset/lm_dataset.py#L58)、[model/model_minimind.py](../model/model_minimind.py#L245)、[tokenizer_config.json](../model/tokenizer_config.json#L333) 和上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L85)。本轮还做了极小 SFT 样本验证：使用本地 tokenizer 与 [SFTDataset](../dataset/lm_dataset.py#L58) 构造一条双轮对话，确认 `user` 区间的 `labels` 会被置为 `-100`，assistant 回复区间会被复制为真实 token id，能够进入 loss。另已用 `diff -u` 核对本地 [dataset/lm_dataset.py](../dataset/lm_dataset.py#L58) 与上游 [dataset/lm_dataset.py](../../../references/minimind/dataset/lm_dataset.py#L58)、本地 [model/model_minimind.py](../model/model_minimind.py#L245) 与上游 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py#L245) 在 SFT 相关主逻辑上本轮无差异。
- 当前本地代码事实：
  当前本地仓库已经有 [SFTDataset](../dataset/lm_dataset.py#L58)、chat template 配置 [tokenizer_config.json](../model/tokenizer_config.json#L333) 和语言模型 loss 计算 [model/model_minimind.py](../model/model_minimind.py#L250)。但本地 `trainer/` 目录目前只有 `train_pretrain.py` 与 `trainer_utils.py`，没有同步 `train_full_sft.py` 这个训练入口。
- 上游 MiniMind 事实：
  上游 SFT 训练入口在 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L85)。它默认读取 [SFTDataset](../../../references/minimind/dataset/lm_dataset.py#L58)，默认 `data_path="../dataset/sft_t2t_mini.jsonl"`、`from_weight="pretrain"`、`learning_rate=1e-5`，训练时仍然调用 `model(input_ids, labels=labels)`，损失仍由 [model/model_minimind.py](../../../references/minimind/model/model_minimind.py#L250) 内部的 shifted cross entropy 计算。
- 工程判断或后续计划：
  本轮没有实际跑完整 SFT 训练，没有产出 `full_sft` 权重，没有验证真实收敛曲线、回答质量、吞吐或显存峰值。关于“RTX 5060 Laptop 约 8GB 显存下应该怎样缩配置”这一类内容，下文只给工程建议，不写成本轮已实测结论。

## 可直接口述回答（>=1000字）

如果让我从 0 开始解释什么是 SFT，我会先给一句最短的结论：SFT 是 `Supervised Fine-Tuning`，中文通常叫“监督微调”。它的意思不是重新发明一个新模型，也不是换一种完全不同的训练数学，而是拿一个已经会“续写文字”的基础模型，再用“输入长什么样、理想回答应该长什么样”的样本，进一步把它往你想要的回答风格和任务格式上推一把。

可以先把预训练和 SFT 分开理解。预训练更像“让模型先大量阅读”，它学会的是很宽泛的语言规律，比如一个词后面大概率跟什么词，一段句子通常怎么接下去。而 SFT 更像“老师拿着标准题和参考答案，专门教它在某种场景下应该怎么答”。如果说预训练是在教一个人识字、读书、模仿语言；那 SFT 更像是在做岗位培训，告诉他“当用户这样提问时，你要站在 assistant 角色这样回答，而不是乱续写或者切错角色”。

MiniMind 里这件事不是一句抽象口号，而是一条可以落到代码的链路：

$$
\text{tokenizer} \rightarrow \text{SFTDataset} \rightarrow \text{model} \rightarrow \text{loss} \rightarrow \text{backward} \rightarrow \text{optimizer} \rightarrow \text{checkpoint} \rightarrow \text{generate}
$$

这条链里，最关键的不是“名字从 pretrain 换成了 sft”，而是监督信号变了。MiniMind 的 SFT 数据在 [SFTDataset](../dataset/lm_dataset.py#L58) 里处理。它先把 `conversations` 这种多轮对话数据，通过 [create_chat_prompt](../dataset/lm_dataset.py#L71) 和 [tokenizer_config.json](../model/tokenizer_config.json#L333) 里的 `chat_template`，拼成模型真正看到的 token 序列。然后 [generate_labels](../dataset/lm_dataset.py#L88) 不会像预训练那样“整段文本几乎都参与 loss”，而是只把 assistant 回复区间对应的 label 变成真实 token id，其他位置一律写成 `-100`。

这里的 `-100` 很重要。它不是词表里的 token，也不是 tokenizer 编出来的 id。它只是 PyTorch 里 `F.cross_entropy(..., ignore_index=-100)` 使用的一个忽略标记。谁的 label 是 `-100`，谁就不计入损失。所以 SFT 的重点并不是“模型只看 assistant 内容”，而是“模型仍然会读完整个 prompt，但只在 assistant 应该说的话那一段上挨老师批改”。这和人做口语考试很像：考官会把题目完整读给你听，但最后给分的不是考官提问那段，而是你自己的回答那段。

MiniMind 当前模型端的 loss 在 [model/model_minimind.py](../model/model_minimind.py#L250) 到 [model/model_minimind.py](../model/model_minimind.py#L252)：

$$
x = logits[..., :-1, :]
$$

$$
y = labels[..., 1:]
$$

$$
\mathcal{L}_{\text{SFT}} = \text{CrossEntropy}(x, y, \text{ignore\_index}=-100)
$$

这说明 SFT 在 MiniMind 里仍然是标准的“下一个 token 预测”。它没有把语言模型目标换掉，而是通过 `labels` 的掩码规则，把“哪些 token 该被监督”改掉了。更完整一点可以写成：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|S|}\sum_{(b,t)\in S}\log p_{\theta}(y_{b,t+1}\mid x_{b,\le t})
$$

这里：

- $b$ 表示 batch 里的第几条样本。
- $t$ 表示序列里的位置。
- $x_{b,\le t}$ 表示当前位置之前已经看到的所有 token。
- $y_{b,t+1}$ 表示下一个位置的真实 token。
- $S$ 表示“有效监督位置的集合”，也就是那些 `labels[b, t+1] != -100` 的地方。
- $\theta$ 表示模型参数。

这条公式背后的人话是：模型每次都根据前面的上下文，去猜下一个 token 是什么；但是只有 assistant 回复区间里的“下一个 token”猜错了，老师才扣分。用户问题、系统提示、padding 这些地方虽然也进入 `input_ids`，但不会直接记分。

这件事在本轮本机极小验证里可以看得很直观。我用本地 [SFTDataset](../dataset/lm_dataset.py#L58) 和 tokenizer 做了一条双轮对话样本，观察 `input_ids` 与 `labels` 的对齐关系。结果显示，前面的 `user` 区间 label 全部是 `-100`；assistant 开始回答之后，label 才逐步变成真实 token id。换句话说，SFT 并不是训练模型“复述用户问题”，而是训练模型“看到用户问题和 assistant 起始前缀后，接下来应该怎样回答”。

一个容易卡住的点是：既然 assistant 起始前缀本身通常也被 mask 掉，那模型怎么学会“开始回答”的第一个 token 呢？答案在 shifted loss。假设 assistant 真正内容的第一个 token 在位置 17，那么训练时参与对比的是 `logits[16]` 和 `labels[17]`。也就是说，模型看完“前面的用户问题 + assistant 起始头部”以后，被要求预测 assistant 的第一个真实内容 token。这恰恰就是对话模型最关键的能力之一。

SFT 和预训练还有一个非常重要的差别：训练数据结构不一样。预训练样本往往只是一条普通文本，比如 `{"text": "今天天气很好"}`。SFT 样本则更像“消息列表”，里面会有 `system`、`user`、`assistant`，有时还有 `tool`、`tool_calls`。MiniMind 的 [create_chat_prompt](../dataset/lm_dataset.py#L71) 就是在做这件事：把原始消息结构展开成模型真实能吃的文本模板。这一步决定了模型以后推理时为什么也要尽量复现相似模板，否则它训练时学到的“轮到 assistant 回答”的信号就会弱很多。

再说训练主循环。虽然本地仓库目前没同步 `train_full_sft.py`，但上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L136) 非常直白：先加载 `SFTDataset`，再把 batch 送进 `model(input_ids, labels=labels)`，得到 `loss` 后做 `backward`，再让 optimizer 更新参数。参数更新可以抽象写成：

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}
$$

这里 $\eta$ 是学习率，$\nabla_{\theta}\mathcal{L}$ 是损失对参数的梯度。人话就是：如果模型这次回答偏了，梯度会告诉参数“应该往哪个方向改”，学习率决定“这次改多大一步”。

为什么上游默认 `from_weight="pretrain"`？因为 SFT 通常不是从零开始教模型说话，而是站在已经会基本语言规律的预训练模型上，再去教它更像一个对话助手。如果直接从随机初始化开始跑 SFT，就像一个连语言基本规律都没学会的人，先上岗做客服，效率会很差。

最后要把边界说清楚。本轮我能确认的是：MiniMind 当前代码里，SFT 的核心机制已经很明确，数据模板、label mask、shifted cross entropy 这条链条都能在本地文件里落点，并通过极小样本验证到“assistant 才是主要监督对象”。但我没有在本轮实际跑完整 SFT，也没有验证真实对话效果提升，更没有得到本机上的 loss 曲线。所以更稳妥的表述应该是：**我已经读懂并验证了 MiniMind 的 SFT 数据与 loss 语义；完整训练效果和显存表现还需要后续 smoke test 与正式实验确认。**

## 详细原理讲解（>=3000字，含公式）

### 1. 先把词拆开：S、F、T 分别是什么意思

很多人第一次听到 SFT，会把它当成一个“高级名词”。其实把三个词拆开，反而更容易理解。

- `Supervised`：监督。
- `Fine-Tuning`：微调。

“监督”这两个字，在机器学习里最朴素的含义就是：你不只是给模型看输入，还同时告诉它“正确答案应该是什么”。比如给它一道题，再给它标准答案。模型不是靠自己瞎猜后没人管，而是每一步都会被拿去和标准答案比较，错了就改参数。

“微调”表示：不是从零开始造一个全新的模型，而是在已有模型参数上继续训练，让它更适应某一种任务或输出风格。

所以 SFT 合在一起，就是：**拿一个已经有基础能力的模型，再用“输入 + 标准回答”的样本，把它往特定任务方向继续训练。**

### 2. 为什么预训练之后还需要 SFT

如果只做预训练，模型学到的核心能力是“语言续写”。这很强，但还不够像一个对话助手。因为“会续写”不等于“会扮演 assistant，稳定按指令回答问题”。

可以用一个非常生活化的比喻。假设有两个人：

- A 读了很多书，语感很好，能自然接下一句话。
- B 不但读了很多书，还专门做过客服岗位培训，知道什么时候要礼貌回应、什么时候该直接回答、什么时候不能跑题。

预训练更像把 A 培养出来，SFT 更像把 A 再培训成 B。

再用一个更接近 MiniMind 的比喻。预训练像让模型看过海量“自然文本”；SFT 像老师给它一本“问答规范手册”，里面写着：

- 当看到 `user` 的问题时，接下来应该由 `assistant` 说话。
- 用户的问题本身不是要你复述的标准答案。
- 你的回答要落在 assistant 段落，而不是把整段对话模板乱续写。

所以 SFT 的价值，不只是让模型“知道更多事实”，更关键的是让模型“在某种输入格式下，学会更像样地回答”。

### 3. 从 MiniMind 代码看，SFT 真正发生在哪一层

很多初学者会误以为“只要有一个 `train_full_sft.py` 文件，SFT 就神秘发生了”。实际不是。SFT 的关键不在文件名，而在监督信号怎样构造。

在 MiniMind 里，这条主线可以拆成七步：

1. 原始样本不是纯文本，而是 `conversations`。
2. [create_chat_prompt](../dataset/lm_dataset.py#L71) 调用 tokenizer 的 chat template，把多轮消息拼成一整段字符串。
3. 这段字符串再被 tokenizer 编码成 `input_ids`。
4. [generate_labels](../dataset/lm_dataset.py#L88) 根据 assistant 边界，生成 `labels`。
5. `input_ids` 和 `labels` 被送进 [MiniMindForCausalLM.forward](../model/model_minimind.py#L245)。
6. 模型输出 `logits`，并在 [model/model_minimind.py](../model/model_minimind.py#L251) 用 shifted cross entropy 算 loss。
7. 上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L24) 到 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L47) 做 `backward`、梯度裁剪、`optimizer.step()` 和 checkpoint 保存。

你会发现：**SFT 本质上是“数据模板 + label mask + 同一个自回归 loss”的组合**。模型结构没有因为 SFT 换掉，Causal LM 目标也没有因为 SFT 换掉；真正变的是“哪些 token 要被计入监督”。

### 4. tokenizer 和 chat template 在 SFT 中为什么这么重要

这是很多新手最容易低估的一步。你可能会觉得：“不就是把用户问题和答案拼起来吗？” 但对模型来说，怎么拼，影响非常大。

当前本地 [tokenizer_config.json](../model/tokenizer_config.json#L333) 里定义了 `chat_template`。它会把多轮消息拼成类似下面这种结构：

```text
<|im_start|>user
1加1等于几？<|im_end|>
<|im_start|>assistant
1加1等于2。<|im_end|>
```

如果存在思考标签、工具调用、system 消息，模板还会更复杂。

为什么这很重要？因为模型不是直接理解“这是一条用户消息，那是一条回答消息”这种抽象概念。它真正看到的是 token 序列。谁是 `user`，谁是 `assistant`，什么时候该开始回答，什么时候该结束，最终都要体现在 token 模板里。

换句话说，chat template 就像舞台剧剧本里的“人物名标识”。没有它，模型只看到一堆连续文本，不一定知道当前轮到谁说话。有了它，模型在 SFT 中就能反复学到：

- 看到 `user` 段落后，通常接 `assistant` 段落。
- 进入 `assistant` 段落后，后续 token 才是主要监督对象。
- `<|im_end|>` 是一段消息结束的信号。

### 5. labels 到底是什么，为什么不是所有 token 都要参与 loss

先说最核心的一点：在 MiniMind 里，`input_ids` 是模型实际看到的输入；`labels` 是老师拿来打分的答案标记。

预训练时，`labels` 往往接近 `input_ids` 的拷贝，只是 padding 位置改成 `-100`。但 SFT 不一样。SFT 要做的是“让模型回答 assistant 内容”，不是“让模型复述整段模板”。

因此 [generate_labels](../dataset/lm_dataset.py#L88) 的逻辑是：

1. 先把整段 `labels` 初始化成 `-100`。
2. 找到形如 `<|im_start|>assistant\n` 的起点模式。
3. 再找到对应的 `<|im_end|>\n` 终点模式。
4. 把起点和终点之间那一段的 label 改成真实 token id。

它并没有把整段 `input_ids` 都复制到 `labels`，而是只复制 assistant 回复相关的那部分。

这一步可以用一个非常小的表来理解：

| 位置类型 | 会不会进入 `input_ids` | `labels` 是什么 | 是否参与 loss |
| --- | --- | --- | --- |
| `user` 角色头和用户问题 | 会 | `-100` | 不参与 |
| `assistant` 角色头 `<|im_start|>assistant\n` | 会 | 通常是 `-100` | 不参与 |
| assistant 真正回复内容 | 会 | 复制为真实 token id | 参与 |
| assistant 结束标记 `<|im_end|>` | 会 | 通常也会被复制 | 参与 |
| padding | 会 | `-100` | 不参与 |

这背后的直觉是：模型需要“看见题目”，但只在“该回答的部分”被打分。

### 6. 一个最小数字案例：为什么 `-100` 不是 token，而是“这题不判分”

假设词表只有 3 个 token：`A`、`B`、`C`。某个位置上，模型给出的概率是：

$$
p(A)=0.2,\quad p(B)=0.7,\quad p(C)=0.1
$$

如果正确答案是 `B`，那么单位置交叉熵是：

$$
L=-\log 0.7 \approx 0.357
$$

这个损失不大，说明它答得不错。

如果另一个位置模型给出的概率是：

$$
p(A)=0.8,\quad p(B)=0.1,\quad p(C)=0.1
$$

而正确答案仍然是 `B`，那么：

$$
L=-\log 0.1 \approx 2.303
$$

损失就明显更大，说明它答偏了。

那 `-100` 是什么？它不是第四个 token，也不是“错误答案”。它的含义更像“这一题老师不判分”。在 PyTorch 的 `ignore_index=-100` 规则下，这个位置直接不进入平均，不会贡献正负奖励。

所以 SFT 的威力来自两个动作：

- 给 assistant 回复位置明确标准答案。
- 给不该评分的位置明确“不判分”。

### 7. 再看一个更贴近 MiniMind 的案例：为什么模型能学会“开始回答的第一个字”

假设某条样本 token 位置简化后是：

| 下标 | token |
| --- | --- |
| 0 | `<|im_start|>` |
| 1 | `user` |
| 2 | `\n` |
| 3 | `1加1等于几？` |
| 4 | `<|im_end|>` |
| 5 | `\n` |
| 6 | `<|im_start|>` |
| 7 | `assistant` |
| 8 | `\n` |
| 9 | `1` |
| 10 | `加` |
| 11 | `1` |
| 12 | `等于` |
| 13 | `2` |
| 14 | `。` |
| 15 | `<|im_end|>` |

SFT 的 `labels` 可能长成这样：

| 下标 | `labels` |
| --- | --- |
| 0 到 8 | `-100` |
| 9 | `1` |
| 10 | `加` |
| 11 | `1` |
| 12 | `等于` |
| 13 | `2` |
| 14 | `。` |
| 15 | `<|im_end|>` |

然后模型端做 shift：

$$
x = logits[..., :-1, :]
$$

$$
y = labels[..., 1:]
$$

这意味着：

- `logits[8]` 去预测 `labels[9]`，也就是第一个回答 token `1`。
- `logits[9]` 去预测 `labels[10]`，也就是 `加`。
- `logits[10]` 去预测 `labels[11]`，也就是下一个 `1`。

这就是为什么即便 assistant 头部本身没被直接监督，模型依然能学会“看到 assistant 头部后，开始输出第一个回答 token”。

### 8. MiniMind 的 SFT 损失公式到底和预训练有什么同与不同

相同点：

- 都是自回归语言模型。
- 都是用 `logits[..., :-1, :]` 对 `labels[..., 1:]`。
- 都是 `F.cross_entropy(..., ignore_index=-100)`。

不同点：

- 预训练里，大量普通文本 token 会参与监督。
- SFT 里，主要只有 assistant 回复区间参与监督。
- 预训练更关注广泛语言规律。
- SFT 更关注“在对话模板下怎么回答”。

如果要用集合写法表达这种差别，可以写成：

$$
S_{\text{pretrain}}=\{(b,t)\mid labels[b,t+1]\neq -100\}
$$

$$
S_{\text{SFT}}=\{(b,t)\mid labels[b,t+1]\neq -100,\ \text{并且该位置属于 assistant 回复区间}\}
$$

你会发现，两者数学外壳很像，真正变化的是有效监督集合 $S$。

### 9. backward、optimizer、checkpoint 在 SFT 里做了什么

如果完全没有深度学习基础，可以把这几步理解成“批改作业后改脑子里的参数”。

1. `forward`：
   模型先根据当前参数，给每个位置输出对词表的打分，也就是 `logits`。
2. `loss`：
   把打分和正确答案比较，得到“错得有多离谱”的数字。
3. `backward`：
   反向传播计算“每个参数往哪个方向改，能让损失下降”。
4. `optimizer.step()`：
   真正把参数改一下。
5. `checkpoint`：
   把当前模型参数和优化器状态保存下来，以便中断后继续。

参数更新的抽象公式是：

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}
$$

这里：

- $\theta$ 是模型所有参数。
- $\eta$ 是学习率。
- $\nabla_{\theta}\mathcal{L}$ 是损失对参数的梯度。

如果把模型想象成一个反复被老师纠正的学生，`loss` 是这次答题错了多少，`gradient` 是老师告诉他“哪个知识点该往哪边修”，`optimizer` 则是真正执行修改的过程。

上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L24) 到 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L81) 的训练循环，核心就是在做这件事。它和预训练最大的差别并不是训练循环本身，而是 batch 里装的 `labels` 已经换成了 SFT 语义。

### 10. SFT 和推理 `generate` 是什么关系

这也是很容易混淆的一点。训练时模型有标准答案，所以它能算 loss；推理时没有标准答案，所以它只能一边生成一边选下一个 token。

但两者共享同一个语言模型头和同一个“当前位置预测下一个 token”的逻辑。SFT 真正改变的是：它把参数调到更适合“聊天模板 -> assistant 回答”的分布上。

所以你可以这样理解：

- 预训练让模型“会说话”。
- SFT 让模型“更会按要求回答”。
- `generate` 只是把当前参数拿来实际出答案。

这也是为什么上游 [eval_llm.py](../../../references/minimind/eval_llm.py#L36) 默认常用 `full_sft` 权重做对话推理。因为 SFT 后的模型更熟悉 chat template，更知道 assistant 段落应该怎样续写。

### 11. MiniMind 当前实现里一个很容易忽略的细节：SFT 数据预处理带随机性

本地 [pre_processing_chat](../dataset/lm_dataset.py#L9) 和 [post_processing_chat](../dataset/lm_dataset.py#L31) 都包含随机逻辑：

- 可能以一定概率补一个 system prompt。
- 可能以一定概率移除空的 `<think>\n\n</think>\n\n`。

这意味着：同一条原始 `conversations`，在不同随机种子下，最终 token 序列可能略有差别，assistant 回复区间的精确边界也会跟着变化。

这并不等于 SFT 原理不稳定，而是说明你做最小验证或复现实验时，应该注意随机种子和预处理路径一致性。否则你可能会发现“同样一条样本，这次为什么 `<think>` 还在，上次为什么没了”，进而误以为 label 逻辑错了。

### 12. 另一个容易踩坑的点：`-100` 只屏蔽 loss，不自动屏蔽 attention

这在 MiniMind 项目里值得单独提醒。当前 SFT Dataset 返回的是 `(input_ids, labels)`，上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L36) 也是直接调用：

```python
res = model(input_ids, labels=labels)
```

没有额外把 `attention_mask` 显式传进去。

这意味着：

- padding 位置不会贡献 loss，因为它们在 `labels` 里通常是 `-100`。
- 但 padding token 仍然作为输入 token 进入模型前向。

这件事不能简单说成“有问题”或“没问题”，更准确的说法是：**当前实现主要依赖 label mask，而不是显式 attention mask，padding 对上下文表征的影响还需要后续做最小对照实验确认。** 对初学者来说，最重要的是先记住：`loss mask` 和 `attention mask` 不是一个东西。

### 13. SFT、LoRA、蒸馏是什么关系

如果你刚入门，很容易把这些词搅在一起。可以先这样分：

- SFT：告诉模型“正确回答长什么样”。
- LoRA：不是一种新任务，而是一种更省参数的微调方式。
- 蒸馏：让学生模型模仿教师模型的输出分布。

MiniMind 的路线图里，LoRA 和蒸馏通常都建立在 SFT 基座之后继续展开。原因很简单：如果模型连对话格式都还没被对齐，你先做更复杂的后续训练，往往会把问题放大。

### 14. 在 RTX 5060 Laptop 约 8GB 显存上的现实边界

这部分本轮没有完整实测，所以只能作为工程判断讲。

对 8GB 量级显存来说，全参数 SFT 的真实边界通常比“看文档时想象得更紧”。影响显存的主要变量包括：

- `batch_size`
- `max_seq_len`
- `hidden_size`
- `num_hidden_layers`
- dtype 是 `bf16` 还是 `fp16`
- 是否开启梯度累积
- 是否保存优化器状态

更稳妥的入门顺序通常是：

1. 先只验证单条样本的 `SFTDataset` 输出对不对。
2. 再做单 batch 前向和 loss。
3. 再做单 batch backward。
4. 最后才尝试极小步数的 full SFT smoke test。

如果一开始就把 `batch_size`、`max_seq_len` 和层数开得太大，很容易 OOM，而你还分不清问题是数据错了、loss 错了，还是单纯显存不够。

### 15. 从项目表达角度，哪些话能说，哪些话暂时不能说

当前比较稳妥、能写进实验记录或面试复盘的话是：

- 我已经读懂 MiniMind 的 SFT 数据构造、chat template、label mask 和 shifted cross entropy 关系。
- 我已用极小样本验证过 `user` 区间不参与 loss、assistant 回复区间参与 loss。
- 我知道上游 SFT 训练入口如何从 `pretrain` 权重继续训练。

当前不该直接写成既成事实的话是：

- 我已经完成了 MiniMind 的全量 SFT 训练。
- 我已经验证了 full SFT 在本机上的收敛效果和显存表现。
- 我已经证明当前 SFT 配置优于上游或优于其他方法。

两者的差别，不在于谦虚，而在于证据边界。

## 项目落地点

### 已实现或可直接落到当前本地仓库的部分

- 数据模板与标签构造：
  当前本地 [SFTDataset](../dataset/lm_dataset.py#L58) 已实现 SFT 数据读取、chat template 展开、assistant 区间标签生成。
- 预处理细节：
  当前本地 [pre_processing_chat](../dataset/lm_dataset.py#L9) 与 [post_processing_chat](../dataset/lm_dataset.py#L31) 已实现 system 注入与空 `<think>` 标签处理。
- chat template 来源：
  当前本地 [tokenizer_config.json](../model/tokenizer_config.json#L333) 已定义 `system/user/assistant/tool` 模板。
- 语言模型损失：
  当前本地 [MiniMindForCausalLM.forward](../model/model_minimind.py#L245) 已实现 shifted cross entropy，`ignore_index=-100` 在 [model/model_minimind.py](../model/model_minimind.py#L252) 生效。

### 上游引用路径，当前本地尚未同步为训练入口的部分

- SFT 训练主入口：
  上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L85)。
- SFT 默认训练数据路径与默认权重起点：
  上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L102) 到 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L103)。
- SFT 主训练循环与保存逻辑：
  上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L24) 到 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L81)。

### 已验证、需要验证、后续可扩展

- 本机已验证：
  已验证 SFTDataset 的 `labels` 不是整段复制，而是只对 assistant 回复区间给真实 token id。
- 需要验证：
  需要做真正的 single-batch SFT 前向、loss、backward、参数更新闭环；需要验证 padding 未显式传 `attention_mask` 时的实际影响；需要验证本机 8GB 显存下的可跑配置。
- 后续可扩展：
  在 SFT 闭环稳定后，再扩到 LoRA、蒸馏，以及更复杂的偏好优化分支。

### 哪些内容能写进 README、实验记录或面试材料

- 可以写：
  “已读懂并最小验证 MiniMind 的 SFT 数据与 loss 语义，能够解释 chat template、assistant label mask、shifted cross entropy 和 `from_weight='pretrain'` 的作用。”
- 暂时不要写：
  “已完成 full SFT 训练并验证效果提升。”

## 面试官 / 评审者可能追问与回答

### 追问 1：SFT 和预训练的 loss 公式是不是完全不同？

更准确的回答是：在 MiniMind 当前实现里，外层数学形式没有本质变化，仍然是 [model/model_minimind.py](../model/model_minimind.py#L251) 的 shifted cross entropy；真正变化的是 `labels` 的构造。预训练大部分普通文本 token 会参与监督，SFT 则主要只让 assistant 回复区间参与监督。所以如果要保守表述，应说“**目标函数框架相同，监督集合不同**”，而不是笼统说“换了一套完全不同的 loss”。

### 追问 2：为什么不让用户问题也参与 loss？

因为 SFT 的目标通常不是教模型复述用户问题，而是教模型“在看到问题后如何回答”。MiniMind 的 [generate_labels](../dataset/lm_dataset.py#L88) 就是在实现这个边界。如果把 user 段也全算进 loss，模型会被鼓励去记忆整段模板和用户输入本身，监督重点会变得不够聚焦。更保守的说法是：当前 MiniMind SFT 路线选择了“assistant-only supervision”，这是当前代码事实，不要泛化成“所有 SFT 都必须如此”。

### 追问 3：`-100` 是不是一个特殊 token？

不是。它不是 tokenizer 词表里的 token id，而是交叉熵的忽略标记。当前证据在 [model/model_minimind.py](../model/model_minimind.py#L252)。如果把它说成“模型会生成 `-100`”，那就是错误表述。更准确的说法是：`-100` 只存在于训练标签里，用来告诉 loss 哪些位置不判分。

### 追问 4：为什么 SFT 默认要从 `pretrain` 权重开始？

因为 SFT 本质上是“在已有语言能力上继续对齐”，而不是从零训练对话模型。上游 [trainer/train_full_sft.py](../../../references/minimind/trainer/train_full_sft.py#L103) 默认 `from_weight='pretrain'` 就体现了这一点。面试里不要夸大成“我已经验证从 pretrain 起步一定最好”；更稳妥的说法是“**当前 MiniMind 上游默认这样设计，工程动机是先继承基础语言能力，再做对话对齐**”。

### 追问 5：你怎么证明自己不是只会背概念，而是真的看过代码？

一个有证据的回答是：我能明确指出 MiniMind 里 SFT 的关键落点在 [SFTDataset](../dataset/lm_dataset.py#L58)、[generate_labels](../dataset/lm_dataset.py#L88)、[chat_template](../model/tokenizer_config.json#L333) 和 [MiniMindForCausalLM.forward](../model/model_minimind.py#L245)。我还能解释为什么 `labels[assistant内容]=token_id` 与 `logits[..., :-1]` / `labels[..., 1:]` 的 shift 配合后，会让模型学会在 assistant 头部之后开始回答。当前我已经做过极小样本验证，但还没做完整 full SFT 收敛实验，所以不会把“读懂数据与 loss 链路”包装成“已经完成 full SFT 训练”。

## 最小验证命令示例

下面这类命令适合后续继续复核 SFT 数据语义。它不是本轮生成的正式测试文件，只是便于复现实验思路：

```bash
cd /home/harry/projects/MiniMind
PYTHONPATH=/home/harry/projects/MiniMind direnv exec . python - <<'PY'
import json
import tempfile
from pathlib import Path
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

tokenizer = AutoTokenizer.from_pretrained("./model")
sample = {
    "conversations": [
        {"role": "user", "content": "1加1等于几？", "reasoning_content": "", "tools": "", "tool_calls": ""},
        {"role": "assistant", "content": "1加1等于2。", "reasoning_content": "", "tools": "", "tool_calls": ""}
    ]
}

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "tiny.jsonl"
    path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    ds = SFTDataset(str(path), tokenizer, max_length=128)
    input_ids, labels = ds[0]
    print("有效监督 token 数:", int((labels != -100).sum()))
    print("前 20 个 labels:", labels[:20].tolist())
PY
```

如果继续往前推进，下一步最有价值的不是立刻长时间训练，而是把这类最小验证沉淀成 `tests/test_dataset_labels.py` 或单 batch SFT smoke test。
