"""Semantic ID 前缀树。

根据全部真实物品的 semantic token 序列建立前缀约束，推理时查询某个
已生成前缀允许连接的下一批 token，避免 Beam Search 生成不存在的物品 ID。
"""

from collections import defaultdict

from tiger_min.tiger.tokenizer import TigerTokenizer


class SemanticIdTrie:
    """Store valid next tokens for every semantic-token prefix."""

    def __init__(self, tokenizer: TigerTokenizer) -> None:
        children: dict[tuple[int, ...], set[int]] = defaultdict(set)

        for item_id in range(tokenizer.num_items):
            tokens = tokenizer.item_to_semantic_tokens(item_id)
            for position, token in enumerate(tokens):
                # Every prefix only points to tokens used by at least one real item.
                children[tuple(tokens[:position])].add(token)

        self._children = {
            prefix: tuple(sorted(next_tokens))
            for prefix, next_tokens in children.items()
        }
        self.semantic_id_length = tokenizer.semantic_id_length

    def allowed_next_tokens(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        """Return valid global token ids after ``prefix``."""
        if len(prefix) >= self.semantic_id_length:
            return ()
        return self._children.get(prefix, ())
