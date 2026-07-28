# Beam Search 推理

TIGER 推理阶段没有真实 target。模型只能根据用户历史，从 `BOS` token 开始自回归生成目标物品的 semantic ID。

## 为什么需要 Beam Search

如果每一步只选择概率最大的 token，就是 greedy decoding。它速度快，但容易在早期选择错误 token 后无法恢复。

Beam Search 会在每一步保留多条高概率路径。例如 `beam_size=50` 时，模型每生成一个 semantic token 都保留累计概率最高的 50 条候选序列。生成完成后，再把候选 semantic ID 映射回真实 item ID，得到 Top-K 推荐。

普通 Beam Search 虽然能限制每个位置的 token 范围，但不同位置的合法 token 任意组合后，不一定对应真实物品。为此，本项目增加 Trie 前缀约束，只扩展能够组成真实 semantic ID 的路径。

## 本项目实现

Trie 位于 [tiger_min/tiger/trie.py](../tiger_min/tiger/trie.py)，批量解码和评估位于 [tiger_min/tiger/inference.py](../tiger_min/tiger/inference.py)。

推理流程：

```text
history items
-> tokenizer converts history to encoder tokens
-> Transformer encoder runs once
-> decoder starts from BOS
-> query Trie with every generated prefix
-> mask tokens that cannot continue to a real item
-> beam search keeps the highest-scoring valid paths
-> semantic ID candidates
-> map back to item IDs
-> remove duplicate items
-> Top-K recommendations
```

为了减少重复计算，本项目会缓存 encoder memory。Beam Search 的每一步只重复运行 decoder。

## Token 范围限制

本项目的 tokenizer 为不同 semantic ID 位置分配不同 token 区间。例如：

```text
PAD = 0
BOS = 1
EOS = 2

position 0: codebook offset 3
position 1: codebook offset 3 + codebook_size
position 2: codebook offset 3 + 2 * codebook_size
suffix:     final suffix offset
```

因此推理时每一步先截取当前位置对应的 token 区间，再叠加 Trie 的前缀约束。

## Trie 前缀约束

构建 Trie 时，将每个真实物品的完整 semantic token 序列依次插入前缀树。例如存在：

```text
[15, 304, 522, 771]
[15, 304, 540, 772]
```

那么空前缀只能选择 `15`，前缀 `[15]` 只能选择 `304`，前缀 `[15, 304]` 只能选择 `522` 或 `540`。模型仍然负责给合法 token 计算概率，Trie 只负责排除不可能映射到真实物品的路径。

每一轮解码流程如下：

1. Decoder 根据用户历史和当前生成前缀输出下一 token 的 logits。
2. 根据前缀查询 Trie，得到允许的下一 token 集合。
3. 将其他 token 的 logits 设为负无穷。
4. 对合法候选计算累计对数概率并执行 `topk`。
5. 重复上述过程，直到生成完整 semantic ID。

## 当前实现取舍

当前实现会在每轮解码时为 batch 中的 beam 动态构造合法 token mask，逻辑直观且便于检查。后续可以把 Trie 状态转移预编码成张量，进一步降低大 beam 推理时的 CPU 调度开销。
