# Electra Pretrain

在哈工大训练的electra基础上使用领域数据继续进行预训练，一般能够提升下游任务效果。

## 改动

- 由于我们的语料是单句粒度，修改数据构建方法，只构建单句的语料
- 针对中文，使用更简单的tokenizer，即将所有字符直接拆分【主要是适配下游的NER任务】
- 修改预训练代码，支持加载预训练的模型的参数

## 使用

新建个DATA_DIR，然后在里面新建texts目录，将文本数据放入。

需要根据自己的语料，修改configure_pretraining的参数，包括max_seq_len，num_train_steps等。

运行pretrain.sh【根据自己的实际场景修改参数】。

> 建议自己阅读run_pretrain.py的代码，理解里面的各种参数配置。

## 效果

在ccks2021-track2赛道上进行了测试，用track2和track3的数据继续预训练electra-base，训练33k步后，指标为：

```python
disc_accuracy = 0.96376425
disc_auc = 0.97588205
disc_loss = 0.11515158
disc_precision = 0.79076445
disc_recall = 0.32165003
global_step = 33000
loss = 6.575825
masked_lm_accuracy = 0.7298883
masked_lm_loss = 1.2599187
sampled_masked_lm_accuracy = 0.6684708
```

在track2这个NER任务上，直接使用中文的electra-base模型，dev的F1指标为94.19，继续预训练后可以提升到94.74【线上为92.567，单模型】。