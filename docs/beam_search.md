# Beam Search 推理

TIGER 推理阶段没有真实 target。模型只能根据用户历史，从 `BOS` token 开始自回归生成目标物品的 semantic ID。

## 为什么需要 Beam Search

如果每一步只选择概率最大的 token，就是 greedy decoding。它速度快，但容易在早期选择错误 token 后无法恢复。

Beam Search 会在每一步保留多条高概率路径。例如 `beam_size=50` 时，模型每生成一个 semantic token 都保留累计概率最高的 50 条候选序列。生成完成后，再把候选 semantic ID 映射回真实 item ID，得到 Top-K 推荐。

## 本项目实现

当前实现位于 [tiger_min/tiger/inference.py](../tiger_min/tiger/inference.py)。

推理流程：

```text
history items
-> tokenizer converts history to encoder tokens
-> Transformer encoder runs once
-> decoder starts from BOS
-> beam search generates semantic tokens position by position
-> semantic ID candidates
-> map back to item IDs
-> filter invalid / duplicate items
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

因此推理时每一步只在当前位置允许的 token 区间内取 top candidates。

## 当前局限

当前 Beam Search 只做位置级 token 范围限制，生成完成后再过滤无效 semantic ID。它还没有实现基于 Trie 的前缀约束搜索。

Trie-constrained Beam Search 可以在生成每一层 code 时根据已经生成的前缀限制下一步 token，只探索真实存在的 semantic ID 前缀。这样通常能减少无效候选，提高 beam 利用率。

这是后续优化方向之一。
