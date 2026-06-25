# TIGER 原理说明

TIGER 的核心思想来自论文 [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)：把推荐任务从“候选物品打分”改写成“目标物品 semantic ID 生成”。

传统推荐系统通常先召回一批候选物品，再用排序模型对候选打分。TIGER 的思路不同：先给每个物品分配一个可生成的离散编号，再训练序列模型根据用户历史生成下一个物品的编号。

## 执行流程

1. 离线阶段为每个物品构建 semantic ID。
2. 训练阶段把用户历史物品和目标物品都查表转成 semantic token。
3. Transformer encoder 接收用户历史 token。
4. Transformer decoder 学习生成目标物品 token。
5. 推理阶段从 `BOS` 开始生成 semantic ID。
6. 将生成出的 semantic ID 反查回真实 item ID。

## 为什么不直接生成原始 item ID

原始 item ID 通常只是数据库编号，不包含语义关系。直接预测原始 ID 等价于在大规模物品集合上做稀疏分类，难以利用物品之间的结构关系。

semantic ID 的目的，是让相似物品共享一部分编号结构。例如两个语义相近的物品可能有相同的前几层 code。这样模型学习的不只是“哪个物品被点击”，也能学习物品编号空间中的局部结构。

## 训练样本形式

推荐样本可以表示为：

```text
history: [item_1, item_2, item_3]
target:  item_4
```

经过 semantic ID 查表后，模型看到的是：

```text
encoder input:
semantic(item_1), semantic(item_2), semantic(item_3)

decoder input:
BOS, token_1(item_4), token_2(item_4), ...

labels:
token_1(item_4), token_2(item_4), ..., EOS
```

训练目标是逐位置预测目标物品的 semantic token。

## 推理流程

推理时没有真实 target，只有用户历史：

```text
history items
-> lookup semantic tokens
-> encoder
-> decoder starts from BOS
-> beam search generates semantic ID candidates
-> lookup item IDs
-> Top-K recommendations
```

这里的关键点是：semantic ID 映射表不是直接输入 Transformer 的模型参数。它负责在 item ID 和 token 序列之间转换。Transformer 接收 token，生成 token，最终结果还需要反查回 item ID。
