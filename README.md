# CCKS2021-赛道二-中文NLP地址要素解析

团队：xueyouluo

初赛：1 - 93.63

复赛：3 - 91.32

> 这里的代码是复赛的全流程代码，需要在32G显存的卡上才能正常跑通，如果没有这么大的显存，可以考虑将seq_length改成32，以及减小batch size。

## 解决方案

### 初赛

整体还是以预训练+finetune的思路，主要在模型结构、预训练、模型泛化能力提升、数据增强、融合、伪标签、后处理等方面做了优化。

#### 模型

现在的实体识别方案很多，包括BERT+CRF的序列标注、基于Span的方法、基于MRC的方法，我这里使用的是基于BERT的Biaffine结构，直接预测文本构成的所有span的类别。相比单纯基于span预测和基于MRC的预测，Biaffine的结构可以同时考虑所有span之间的关系，从而提高预测的准确率。

> Biaffine意思双仿射，如果`W*X`是单仿射的话，`X*W*Y`就是双仿射了。本质上就是输入一个长度为`L`的序列，预测一个`L*L*C`的tensor，预测每个span的类别信息。

具体来说参考了论文[Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2005.07150)，但是稍有区别：

- 纯粹基于bert进行finetune，不利用fasttext、bert等做context embedding抽取，这也是为了简化模型
- 不区分char word的embedding，默认就是char【中文的BERT基本都是char】
- 原来的论文中有上下文的多句话，这里默认都是一句话【数据决定】
- 同时改进了原有greedy的decoding方法，使用基于DAG的动态规划算法找到全局最优解

但是这种方法也有一些局限：

- 对边界判断不是特别准
- 有大量的负样本

> 原来我也实现过[Biaffine-BERT-NER](https://github.com/xueyouluo/Biaffine-BERT-NER),但这里的版本优化了一些。

#### 预训练

在比较了大部分开源的预训练模型后，哈工大的electra效果比较好，因此我们采用了electra的预训练方法。使用了本赛道的所有数据+赛道三的初赛所有数据，构建了预训练样本，分别继续预训练base和large的模型33K步【大概15个epoch】。

> 继续预训练模型可以提升1个百分点左右的效果，还是非常有效的。

#### 泛化能力提升

这些应该是属于比较基本的操作了，主要包括：

- 使用了对抗学习（FGM）的方法，但代价是训练速度慢了一倍
- 在Dropout方面加入了spatial dropout和embedding dropout
- 使用SWA的方法避免局部最优解

需要在验证集上调参找到比较合适的值。

#### 数据增强

我们用到了开源的一份地址解析数据，来自[《Neural Chinese Address Parsing》](https://github.com/leodotnet/neural-chinese-address-parsing)。参考赛道二的标注规范，使用规则将数据进行清洗，并用这份数据作为数据增强的语料。同时利用统计信息稍微优化了一下数据，即认为一个span如果被标注次数大于10，并且有一个类别占比不到10%且标注数量小于5就认为是不合理的并将其抛弃。

我们使用了同类型实体替换的方法进行数据增强，然后将预训练后的模型在这份数据上finetune。最后用赛道本身的数据进行二次finetune。初赛上，上面的流程走下来可以在dev上达到94.71，线上92.56。

#### 融合

融合的提升非常明显。在融合上，我们使用了electra-base和electra-large两个模型，分别进行预训练和finetune，然后5-fold。

最后对实体进行投票，其中base权重1/3，large权重2/3，只选择投票结果大于3的实体作为最终结果。

> 初赛上，base单独5-fold融合为93.0，large单独5-fold融合为93.477。二者加权融合为93.537。

#### 伪标签

在融合的基础上，我们进一步使用了伪标签，即将上面的融合后预测的测试集结果作为伪标签，重新训练了base模型的一个fold，再进行预测，最终线上可以到93.5920。后面我也实验了训练5-fold的模型，测试下来可以到93.6087。

#### 后处理

我这边后处理比较简单，主要对特殊符号进行了处理，由于一些特殊符号在训练集没有见过，导致模型预测错误。对于包含特殊符号的实体，如果特殊符号是在实体的边界，那么直接去除特殊符号，保留原来的实体类型；如果不是，则去除这个实体。在伪标签结果的基础上加后处理，线上到93.6212。

#### 实验结果

| 序号 |                  实验                   | Dev指标 | 线上指标 |
| :--: | :-------------------------------------: | ------- | :------: |
|  1   |         Biaffine + roberta ext          | 92.15   |          |
|  2   |         Biaffine + google bert          | 92.33   |          |
|  3   |                 2 + FGM                 | 92.79   |          |
|  4   | 3 + spatial dropout + embedding dropout | 92.94   |  90.65   |
|  5   |        4 + extra data + finetune        | 93.74   |          |
|  6   |              5 + 数据增强               | 93.98   |          |
|  7   |        6 + roberta ext pretrain         | 94.15   |  92.08   |
|  8   |            5 + electra base             | 94.19   |          |
|  9   |            5 + electra large            | 94.32   |  92.13   |
|  10  |        5 + electra base pretrain        | 94.71   |  92.56   |
|  11  |       5 + electra large pretrain        | 94.54   |          |
|  12  |               10 + 5-fold               | -       |  93.009  |
|  13  |               11 + 5-fold               | -       |  93.499  |
|  14  |                 12 + 13                 | -       |  93.537  |
|  15  |             14 + pseudo tag             | -       |  93.62   |
|  16  |               15 + 5-fold               | -       |  93.63   |

### 复赛

复赛上我对原来的流程基本没有做什么改动【主要也是我也没想到什么好改进的点了】，就是预训练改了一下。

复赛由于线上训练时间12h的限制，我不可能跑那么久的预训练了【我线下训练large的模型花了20多个小时😂】，因此预训练的语料只用了本赛道的数据集+开源的数据集来减少预训练的时间。

> 唯一非常折腾我的是，large模型在复赛的时候效果一直比不上base模型，可能是预训练不够导致的。

我在复赛的时候都是全流程提交的，直接线上调参了。大概的结果如下【都是5-fold】：

| 序号 |           实验           | 线上指标 |
| :--: | :----------------------: | :------: |
|  1   |       Electra-base       |  89.15   |
|  2   |      Electra-large       |  89.58   |
|  3   | Electra-base + pretrain  |  90.74   |
|  4   | Electra-large + pretrain |  90.75   |
|  5   |          3 + 4           |  91.08   |
|  6   |     5 + fake 1-fold      |  91.31   |
|  7   |     6 + fake 5-fold      |  91.32   |

最终复赛的结果就是91.32，离第一还是有2个千分点差距的。更多细节就看代码吧，毕竟全都在代码里面了。

## 运行

### 运行环境

我们选择了英伟达提供的[docker](nvcr.io/nvidia/tensorflow:19.10-py3)作为基础镜像进行训练，主要是为了避免配环境的各种问题。

具体：

- Unbuntu == 16.04
- Python == 3.6.8
- GPU V100 32G
- 1.14.0 <= Tensorflow-gpu <= 1.15.*

### 数据准备

#### 赛道数据

这里不提供比赛的数据，大家自己下载好放在tcdata目录下。

#### 预训练模型

预训练模型我们使用了哈工大开源的[中文ELECTRA模型](https://github.com/ymcui/Chinese-ELECTRA#%E5%A4%A7%E8%AF%AD%E6%96%99%E7%89%88%E6%96%B0%E7%89%88180g%E6%95%B0%E6%8D%AE)，具体为大语料版本的模型：

- [ELECTRA-180g-large, Chinese](https://drive.google.com/file/d/1P9yAuW0-HR7WvZ2r2weTnx3slo6f5u9q/view?usp=sharing)

- [ELECTRA-180g-base, Chinese](https://drive.google.com/file/d/1RlmfBgyEwKVBFagafYvJgyCGuj7cTHfh/view?usp=sharing)

下载后解压在user_data/electra目录下。

#### 额外数据

下载[neural-chinese-address-parsing](https://github.com/leodotnet/neural-chinese-address-parsing)中data目录下train、dev、test数据到user_data/extra_data目录下。

#### 目录结构

```
├── code
│   ├── electra-pretrain
│   └── ...
├── tcdata
│   ├── dev.conll
│   ├── final_test.txt
│   └── train.conll
├── user_data
│   ├── electra
│   │   ├── electra_180g_base
│   │   │   ├── base_discriminator_config.json
│   │   │   ├── base_generator_config.json
│   │   │   ├── electra_180g_base.ckpt.data-00000-of-00001
│   │   │   ├── electra_180g_base.ckpt.index
│   │   │   ├── electra_180g_base.ckpt.meta
│   │   │   └── vocab.txt
│   │   └── electra_180g_large
│   │       ├── electra_180g_large.ckpt.data-00000-of-00001
│   │       ├── electra_180g_large.ckpt.index
│   │       ├── electra_180g_large.ckpt.meta
│   │       ├── large_discriminator_config.json
│   │       ├── large_generator_config.json
│   │       └── vocab.txt
│   ├── extra_data
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── train.txt
│   └── track3 # 这里可以不需要
│       ├── final_test.txt #这是初赛的测试集
│       ├── Xeon3NLP_round1_test_20210524.txt #可以不用，复赛没有使用这个数据
│       └── Xeon3NLP_round1_train_20210524.txt #可以不用，复赛没有使用这个数据
```

### 运行

在code目录下运行

```
sh run.sh
```

具体训练细节参考`pipeline.py`文件。

也有一个简化版本的，把seq_len改成了32，没有5-fold，自己测试跑下来dev上大概为94。

```
sh simple_run.sh
```
