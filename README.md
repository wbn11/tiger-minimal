# TIGER Mini 学习项目

这是一个自己实现的 TIGER 最小学习项目。现在只做第一步：把不同来源的用户行为数据统一成后续模型能使用的序列格式。

## 当前目录

```text
data/
  raw/          原始输入，例如 toy_sequences.json 或 Beauty_5.json.gz
  processed/    程序生成的中间结果

docs/
  data_contract.md   数据格式约定
  step_01_data.md    第一步学习笔记

tests/
  test_data.py       第一步数据流测试

tiger_min/
  data/
    adapters.py      读取 raw / processed 数据
    splits.py        生成 history -> target 样本
    build.py         命令行入口
  utils.py           少量通用工具
```

## 为什么现在没有 configs

配置文件不是必须的。它只是把命令行参数保存成文件，方便重复实验。

你现在还在理解流程，所以先不用配置文件。等你清楚每个参数为什么存在，再把它们整理成 config。

## 当前命令

运行测试：

```powershell
cd E:\TIGER\tiger-minimal
E:\TIGER\tiger-repro\Scripts\python.exe -m unittest discover -s tests
```

处理 toy 数据：

```powershell
cd E:\TIGER\tiger-minimal
E:\TIGER\tiger-repro\Scripts\python.exe -m tiger_min.data.build_sequences
```

这条命令默认等价于：

```powershell
E:\TIGER\tiger-repro\Scripts\python.exe -m tiger_min.data.build_sequences --source processed --input data\raw\toy_sequences.json --output data\processed\toy
```

以后处理 Amazon Beauty 原始数据时：

```powershell
cd E:\TIGER\tiger-minimal
E:\TIGER\tiger-repro\Scripts\python.exe -m tiger_min.data.build_sequences --source raw_amazon --input data\raw\Beauty_5.json.gz --output data\processed\beauty --max-users 5000
```
