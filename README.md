# TIGER Minimal Recommender

**A PyTorch minimal implementation of TIGER-style generative recommendation**

本项目基于 PyTorch 从零实现一条 TIGER 风格的生成式推荐流程，覆盖 item2vec 物品向量、简化 RQ-VAE semantic ID、Tokenizer、Encoder-Decoder Transformer、Beam Search 推理和 HR/NDCG 离线评估。
实验使用 Amazon Beauty 5-core 数据集，处理后包含 22363 个用户、12101 个物品、131413 个训练样本、22363 个验证样本和 22363 个测试样本。
在未使用候选集预筛选的 next-item generation 设置下，当前 Test HR@20 为 0.0557，Test NDCG@20 为 0.0227。
本项目是面向学习和本地复现的非官方最小实现，不追求完全对齐论文原版训练配置或论文指标。

**Repository description:** Minimal TIGER-style generative recommender with item2vec, RQ-VAE semantic IDs, Transformer decoding, beam search, and HR/NDCG evaluation on Amazon Beauty.

**Suggested GitHub Topics:** `recommender-system`, `generative-retrieval`, `tiger`, `rq-vae`, `semantic-id`, `pytorch`, `beam-search`, `amazon-beauty`

## Contents

- [Project Overview](#project-overview)
- [Core Results](#core-results)
- [Workflow](#workflow)
- [Quick Start](#quick-start)
- [Environment And Dataset](#environment-and-dataset)
- [Modules](#modules)
- [Differences From The TIGER Paper](#differences-from-the-tiger-paper)
- [Experiment Configuration And Results](#experiment-configuration-and-results)
- [Limitations And Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Project Overview

TIGER 来自论文 [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)。它的核心思想是把推荐任务从“对候选物品打分”改写成“生成目标物品的 semantic ID”。semantic ID 是物品的离散语义编号，例如一个物品可以表示为 `[12, 5, 98, 3]`。

本项目保留 TIGER 的主干执行流程：

1. 用 item2vec 从用户交互序列学习物品向量。
2. 用简化 RQ-VAE 将物品向量离散化为 semantic ID。
3. 将用户历史和目标物品转换成 semantic token 序列。
4. 用 Encoder-Decoder Transformer 学习根据用户历史生成下一个物品的 semantic ID。
5. 推理时用 Beam Search 生成候选 semantic ID，并反查回真实 item ID。

更详细的原理说明放在：

- [TIGER 原理说明](docs/tiger_principle.md)
- [RQ-VAE 与 semantic ID](docs/rqvae_semantic_id.md)
- [Beam Search 推理](docs/beam_search.md)

## Core Results

当前完整实验使用 Amazon Beauty 5-core 全量处理后数据，评估设置为每个用户预测 1 个 next item。模型直接在全部物品空间中生成推荐结果，没有先用候选召回模块做预筛选，因此任务难度高于“候选集内排序”设置。

| Split | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Valid | 0.0288 | 0.0182 | 0.0482 | 0.0245 | 0.0732 | 0.0308 |
| Test | 0.0200 | 0.0126 | 0.0356 | 0.0176 | 0.0557 | 0.0227 |

| Metric | Value |
| --- | ---: |
| Users | 22363 |
| Items | 12101 |
| Train samples | 131413 |
| Valid samples | 22363 |
| Test samples | 22363 |
| item2vec item coverage | 12068 / 12101 = 99.73% |
| Valid target cold-start ratio | 0.23% |
| Test target cold-start ratio | 0.62% |
| Beam size / Top-K | 50 / 20 |
| Avg. valid predictions | 20.0 |
| Valid prediction rate | 1.0 |

## Workflow

### TIGER Principle

<img src="assets/tiger_principle.svg" alt="TIGER 生成式推荐原理图" width="100%">

### Project Pipeline

<img src="assets/tiger_pipeline.svg" alt="TIGER 最小实现项目流程图" width="100%">

## Quick Start

所有命令都需要在项目根目录执行：

```powershell
cd tiger-minimal
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

准备数据后运行完整流程：

```powershell
python -m tiger_min.data.build_sequences
python -m tiger_min.embedding.train_item2vec
python -m tiger_min.rqvae.train_rqvae --normalize-embeddings
python -m tiger_min.tiger.train_tiger
python -m tiger_min.tiger.inference --split valid
python -m tiger_min.tiger.inference --split test --output data/processed/beauty/tiger/eval_test.json
```

如果要复现实验记录中的 20 轮 TIGER 训练结果：

```powershell
python -m tiger_min.tiger.train_tiger --epochs 20 --output-dir data/processed/beauty/tiger_e20
python -m tiger_min.tiger.inference --checkpoint data/processed/beauty/tiger_e20/tiger.pt --split valid --output data/processed/beauty/tiger_e20/eval_valid_epoch20.json
python -m tiger_min.tiger.inference --checkpoint data/processed/beauty/tiger_e20/tiger.pt --split test --output data/processed/beauty/tiger_e20/eval_test_epoch20.json
```

### Smoke Test

小规模 smoke test 用于快速检查数据流、训练入口和推理入口能否跑通，不用于报告正式指标。

```powershell
python -m tiger_min.data.build_sequences --max-users 5000 --output data/processed/beauty_5k
python -m tiger_min.embedding.train_item2vec --sequences data/processed/beauty_5k/item2vec_sequences.json --output data/processed/beauty_5k/item_embeddings.pt --epochs 2
python -m tiger_min.rqvae.train_rqvae --item-embeddings data/processed/beauty_5k/item_embeddings.pt --output-dir data/processed/beauty_5k --epochs 2 --normalize-embeddings
python -m tiger_min.tiger.train_tiger --splits data/processed/beauty_5k/splits.pt --semantic-ids data/processed/beauty_5k/semantic_ids.pt --output-dir data/processed/beauty_5k/tiger --epochs 1 --max-train-batches 20 --max-valid-batches 5
python -m tiger_min.tiger.inference --checkpoint data/processed/beauty_5k/tiger/tiger.pt --splits data/processed/beauty_5k/splits.pt --split valid --output data/processed/beauty_5k/tiger/eval_valid.json --max-batches 5
```

## Environment And Dataset

### Python Environment

当前复现实验环境：

| Item | Value |
| --- | --- |
| Python | 3.10.8 |
| PyTorch | 2.6.0+cu126 |
| CUDA used by PyTorch | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB |
| NumPy | 2.1.2 |
| scikit-learn | 1.7.2 |
| tqdm | 4.68.3 |

当前 `requirements.txt` 固定为本次实验使用的 CUDA 12.6 PyTorch 环境。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果需要在 CPU 环境运行，可以先按照 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 安装 CPU 版 PyTorch，再安装其余依赖，或按本机环境调整 `requirements.txt` 中的 `torch` 版本。

当前 CUDA 环境对应的 PyTorch 安装方式为：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Dataset

本项目使用 Amazon Reviews 2014 版本中的 Beauty 5-core 数据：

| Item | Value |
| --- | --- |
| Dataset | Amazon Beauty 5-core |
| Source | [Amazon product data, Julian McAuley / UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) |
| File name | `reviews_Beauty_5.json.gz` |
| Local path | `data/raw/reviews_Beauty_5.json.gz` |
| Split | leave-one-out by user sequence |
| Minimum sequence length | 5 |

数据目录结构：

```text
data/
  raw/
    reviews_Beauty_5.json.gz
  processed/
    beauty/
      item2vec_sequences.json
      item2id.json
      user2id.json
      splits.pt
      data_meta.json
      split_meta.json
      item_embeddings.pt
      semantic_ids.pt
      tiger/
```

`data/` 默认不提交 Git。

### Runtime Notes

| Stage | Device | Observed / Status |
| --- | --- | --- |
| Data preprocessing | CPU | TODO: record wall-clock time |
| item2vec | CUDA | TODO: record wall-clock time |
| RQ-VAE | CUDA | TODO: record wall-clock time |
| TIGER training | CUDA | TODO: record wall-clock time |
| Beam evaluation | CUDA | Valid about 2-3 minutes, Test about 3 minutes in the latest run |

## Modules

| Module | Description |
| --- | --- |
| [tiger_min/data/adapters.py](tiger_min/data/adapters.py) | 读取 Amazon 原始数据或已处理序列 |
| [tiger_min/data/splits.py](tiger_min/data/splits.py) | 构建 `history -> target` 样本和 leave-one-out split |
| [tiger_min/data/build_sequences.py](tiger_min/data/build_sequences.py) | 数据处理入口，生成 `splits.pt` 和 item2vec 序列 |
| [tiger_min/embedding/dataset.py](tiger_min/embedding/dataset.py) | item2vec 正样本 pair 和负采样数据集 |
| [tiger_min/embedding/model.py](tiger_min/embedding/model.py) | Skip-Gram Negative Sampling 模型 |
| [tiger_min/embedding/train_item2vec.py](tiger_min/embedding/train_item2vec.py) | item2vec 训练入口 |
| [tiger_min/rqvae/model.py](tiger_min/rqvae/model.py) | 简化 RQ-VAE 和 residual quantizer |
| [tiger_min/rqvae/semantic_id_dedup.py](tiger_min/rqvae/semantic_id_dedup.py) | semantic ID 去重和 suffix 追加 |
| [tiger_min/rqvae/train_rqvae.py](tiger_min/rqvae/train_rqvae.py) | RQ-VAE 训练与 semantic ID 导出 |
| [tiger_min/tiger/tokenizer.py](tiger_min/tiger/tokenizer.py) | item ID、semantic ID 和 Transformer token 之间的转换 |
| [tiger_min/tiger/dataset.py](tiger_min/tiger/dataset.py) | TIGER 训练样本和 batch collate |
| [tiger_min/tiger/model.py](tiger_min/tiger/model.py) | Encoder-Decoder Transformer |
| [tiger_min/tiger/train_tiger.py](tiger_min/tiger/train_tiger.py) | TIGER 训练入口 |
| [tiger_min/tiger/inference.py](tiger_min/tiger/inference.py) | Beam Search 推理和 HR/NDCG 评估 |

## Differences From The TIGER Paper

| Component | TIGER paper | This repository |
| --- | --- | --- |
| Item representation | Content embedding from product text | item2vec from user interaction sequences |
| Item tokenizer | RQ-VAE over content embeddings | Simplified RQ-VAE over item2vec embeddings |
| Semantic ID uniqueness | Full paper system design | Append suffix after RQ-VAE IDs to ensure unique final semantic IDs |
| User input | Historical semantic IDs with full paper setup | Historical semantic IDs only |
| Model implementation | Full training framework and large-scale tuning | Minimal PyTorch `nn.Transformer` implementation |
| Inference | Generative retrieval | Beam Search with post-generation invalid ID filtering |
| Goal | Paper-level benchmark | Local reproducible minimal implementation |

本项目不声称复现论文完整指标。重点是复现 TIGER 的主干思路和工程闭环。

## Experiment Configuration And Results

### Data Statistics

| Metric | Value |
| --- | ---: |
| Users | 22363 |
| Items | 12101 |
| item2vec interactions | 153776 |
| Train samples | 131413 |
| Valid samples | 22363 |
| Test samples | 22363 |

### Configuration

| Stage | Parameter | Value |
| --- | --- | --- |
| Data | min sequence length | 5 |
| Data | max history length | 20 |
| Data | split | leave-one-out |
| item2vec | embedding dim | 256 |
| item2vec | window size | 3 |
| item2vec | negatives per positive | 10 |
| item2vec | batch size | 1024 |
| item2vec | epochs | 15 |
| item2vec | learning rate | 0.005 |
| RQ-VAE | latent dim | 128 |
| RQ-VAE | hidden dim | 256 |
| RQ-VAE | quantizer layers | 3 |
| RQ-VAE | codebook size | 256 |
| RQ-VAE | batch size | 1024 |
| RQ-VAE | epochs | 20 |
| RQ-VAE | learning rate | 0.0005 |
| TIGER | d_model | 192 |
| TIGER | heads | 6 |
| TIGER | encoder layers | 3 |
| TIGER | decoder layers | 3 |
| TIGER | feedforward dim | 512 |
| TIGER | dropout | 0.15 |
| TIGER | batch size | 128 |
| TIGER | epochs | 20 |
| TIGER | learning rate | 0.0003 |
| TIGER | grad clip | 1.0 |
| Beam Search | beam size | 50 |
| Beam Search | top_k | 20 |
| Popular Baseline | ranking rule | global top-K from train-visible item2vec sequences |
| Common | seed | 42 |

### item2vec

| Metric | Value |
| --- | ---: |
| Positive pairs | 654300 |
| Final loss | 0.0748 |
| Unique items in item2vec sequences | 12068 |
| Item coverage | 99.73% |

### RQ-VAE And Semantic ID

| Metric | Value |
| --- | ---: |
| Best epoch | 5 |
| Best loss | 0.0036 |
| Export loss | 0.0036 |
| Base semantic ID length | 3 |
| Final semantic ID length | 4 |
| Unique base semantic IDs | 9969 |
| Base collision rate before suffix | 31.18% |
| Max base collision group size | 7 |
| Final collision rate after suffix | 0 |

说明：`collision_rate = 0.3118` 是追加 suffix 前的原始 RQ-VAE semantic ID 冲突率。最终用于 TIGER 的 `semantic_ids.pt` 已追加 suffix，因此最终 semantic ID 可以唯一映射回物品。

### TIGER Training

| Metric | Value |
| --- | ---: |
| Best epoch | 14 |
| Best valid loss | 1.8335 |
| Final epoch | 20 |
| Final train loss | 1.5542 |
| Final valid loss | 1.8529 |

### Ranking Results

Popular Baseline 指标由下面的程序生成：

```powershell
python -m tiger_min.baselines.popular --processed-dir data/processed/beauty
```

| Model | Split | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TIGER Minimal | Valid | 0.0288 | 0.0182 | 0.0482 | 0.0245 | 0.0732 | 0.0308 |
| TIGER Minimal | Test | 0.0200 | 0.0126 | 0.0356 | 0.0176 | 0.0557 | 0.0227 |
| Popular Baseline | Valid | 0.0098 | 0.0059 | 0.0163 | 0.0079 | 0.0265 | 0.0104 |
| Popular Baseline | Test | 0.0073 | 0.0040 | 0.0114 | 0.0053 | 0.0195 | 0.0073 |
| ItemKNN | Valid | TODO | TODO | TODO | TODO | TODO | TODO |
| ItemKNN | Test | TODO | TODO | TODO | TODO | TODO | TODO |

### Beam Search Validity

Beam Search 有效预测统计由 `tiger_min.tiger.inference` 在评估时输出：

```powershell
python -m tiger_min.tiger.inference --checkpoint data/processed/beauty/tiger_e20/tiger.pt --split test --output data/processed/beauty/tiger_e20/eval_test_epoch20.json
```

| Split | Beam size | Top-K | Avg. valid predictions | Valid prediction rate |
| --- | ---: | ---: | ---: | ---: |
| Valid | 50 | 20 | 20.0 | 1.0 |
| Test | 50 | 20 | 20.0 | 1.0 |

### Target Coverage

item2vec 覆盖率和 valid/test target 冷启动比例由下面的统计程序生成：

```powershell
python -m tiger_min.data.dataset_stats --processed-dir data/processed/beauty
```

| Metric | Value |
| --- | ---: |
| Valid target coverage by item2vec sequence items | 99.77% |
| Test target coverage by item2vec sequence items | 99.38% |
| Valid cold-start target ratio | 0.23% |
| Test cold-start target ratio | 0.62% |

## Limitations And Future Work

- 当前 Beam Search 只按 semantic ID 位置限制 token 范围，生成完成后再过滤无效 semantic ID；尚未实现基于 Trie 的前缀约束搜索。
- 当前 item embedding 来自 item2vec 交互序列，没有接入商品标题、类目、品牌等文本信息。
- 当前已补充 Popular baseline，尚未实现 ItemKNN 等更强序列/共现基线的正式实验结果。
- 当前没有候选集预筛选，模型直接在全部物品空间中生成 next item，任务难度较高。
- 当前训练时间只记录了部分阶段，后续需要系统记录每个阶段 wall-clock time、显存占用和吞吐。
- 后续可以将 item embedding 来源替换为文本 embedding，并比较交互语义和内容语义对 semantic ID 的影响。
- 后续可以实现 Trie-constrained Beam Search，减少无效 semantic ID 候选。

## Citation

If you use the TIGER paper, please cite:

```bibtex
@article{rajput2023recommender,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Keshavan, Raghunandan H. and Vu, Trung and Heldt, Lukasz and Hong, Lichan and Tay, Yi and Tran, Vinh Q. and Samost, Jonah and Kula, Maciej and Chi, Ed H. and Sathiamoorthy, Maheswaran},
  journal={arXiv preprint arXiv:2305.05065},
  year={2023}
}
```

## Acknowledgements

- TIGER paper: [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- Amazon Reviews dataset: [Amazon product data, Julian McAuley / UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- PyTorch: [https://pytorch.org](https://pytorch.org)
