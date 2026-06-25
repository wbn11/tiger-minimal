# RQ-VAE 与 Semantic ID

在 TIGER 中，RQ-VAE 可以理解为物品 tokenizer。它把连续的物品向量转换为离散的 semantic ID，使 Transformer 可以像处理文本 token 一样处理物品。

## 本项目的输入向量

论文原版 TIGER 使用商品文本信息构造物品内容向量。本项目为了保持最小可复现流程，第一版使用 item2vec 从用户交互序列中学习物品向量。

```text
用户交互序列
-> item2vec
-> item_embeddings.pt
-> RQ-VAE
-> semantic_ids.pt
```

后续如果接入商品标题、类目、品牌等文本信息，只要最终输出同样形状的 `item_embeddings.pt`，下游 RQ-VAE 和 TIGER 模型不需要大改。

## 简化 RQ-VAE 结构

本项目实现的 RQ-VAE 结构如下：

```text
item embedding
-> MLP encoder
-> latent vector
-> residual quantization
-> MLP decoder
-> reconstructed embedding
```

训练目标包含：

- reconstruction loss：让重建向量接近原始 item embedding。
- codebook loss：更新码本向量。
- commitment loss：约束 encoder 输出靠近选中的码本向量。

码本使用 KMeans 初始化。sklearn 的 KMeans 在 CPU NumPy 上运行，因此初始化时会把 latent tensor 从 CUDA 转到 CPU NumPy，聚类完成后再把中心转回 PyTorch tensor。

## Semantic ID 去重

RQ-VAE 的多层 code 不一定天然唯一。多个物品可能被量化成相同的 base semantic ID：

```text
item_a -> [12, 5, 98]
item_b -> [12, 5, 98]
```

如果不处理，推理阶段生成 `[12, 5, 98]` 时无法唯一映射回一个物品。

本项目在 base semantic ID 后追加 suffix：

```text
item_a -> [12, 5, 98, 0]
item_b -> [12, 5, 98, 1]
```

实验记录中的 `collision_rate = 0.3118` 指的是追加 suffix 前的原始冲突率。追加 suffix 后，最终用于 TIGER 的 `semantic_ids.pt` 可以唯一映射回 item ID，最终冲突率为 0。

## 输出文件

```text
base_semantic_ids.pt
semantic_ids.pt
semantic_id_meta.json
rqvae.pt
rqvae_train_meta.json
```

- `base_semantic_ids.pt`：RQ-VAE 原始 code。
- `semantic_ids.pt`：追加 suffix 后的最终 semantic ID。
- `semantic_id_meta.json`：冲突率、最大冲突组大小等统计信息。
- `rqvae.pt`：最佳 RQ-VAE checkpoint。
- `rqvae_train_meta.json`：训练过程和导出信息。
