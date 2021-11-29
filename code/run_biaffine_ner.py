#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import *
from collections import defaultdict
from tqdm import tqdm
import time
import random
import tokenization
import optimization


import collections
import os
import sys
import pickle
import json
import pdb

import tensorflow as tf
import numpy as np


import modeling


# 这里为了避免打印重复的日志信息
tf.get_logger().propagate = False

flags = tf.flags

FLAGS = flags.FLAGS

## K-fold
flags.DEFINE_integer("fold_id", 0, "which fold")
flags.DEFINE_integer("fold_num", 1, "total fold number")

flags.DEFINE_integer("seed", 20190525, "random seed")

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_bool("focal_loss", False, "Whether to use focal loss.")
flags.DEFINE_float(
    "neg_sample", 1.0,
    "negative sampling ratio")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_train_and_eval", False,
    "Whether to run training and evaluation."
)
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer(
    "train_batch_size", 64,
    "Total batch size for training.")

flags.DEFINE_integer(
    "eval_batch_size", 32,
    "Total batch size for eval.")

flags.DEFINE_integer(
    "predict_batch_size", 32,
    "Total batch size for predict.")

flags.DEFINE_float(
    "learning_rate", 5e-6,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 10.0,
    "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_bool("horovod", False,
                  "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool(
    "amp", False, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")
flags.DEFINE_string(
    "pooling_type", 'last', "last | first_last "
)

# Dropout
flags.DEFINE_float("embedding_dropout", 0.0, "dropout ratio of embedding")
flags.DEFINE_float("spatial_dropout", 0.0,
                   "dropout ratio of embedding, in channel")
flags.DEFINE_float("bert_dropout", 0.0, "dropout ratio of bert")

# FGM
flags.DEFINE_bool(
    "use_fgm", False,
    "Whether to use FGM to train model.")
flags.DEFINE_float("fgm_epsilon", 0.3, "The epsilon value for FGM")
flags.DEFINE_float("fgm_loss_ratio", 1.0, "The ratio of fgm loss")

flags.DEFINE_float("head_lr_ratio", 1.0, "The ratio of header learning rate")
flags.DEFINE_bool("use_bilstm", False,
                  "Whether to use Bi-LSTM in the last layer.")
flags.DEFINE_bool("extra_pretrain", False,
                  "Whether to use extra data to pretrain model")
flags.DEFINE_bool("enhance_data", False, "Whether to do data enhance")
flags.DEFINE_bool("electra", False, "Whether to use electra")
flags.DEFINE_bool("dp_decode", False, "Whether to use dp to decode")
flags.DEFINE_bool("fake_data", False, "Whether to use fake data")
flags.DEFINE_bool("export_prob", False, "Whether to export span prob")

# SWA
flags.DEFINE_integer("swa_steps", 50, "number of swa step")
flags.DEFINE_integer("start_swa_step", 0, "step to start swa")
flags.DEFINE_integer("biaffine_size", 150, "biaffine size")
flags.DEFINE_integer("enhance_num", 10, "data enhance number")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, raw_text=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.raw_text = raw_text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_mask, gold_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.span_mask = span_mask
        self.gold_labels = gold_labels


def data_enhance(sentences, num=10):
    new_data = []
    # 获取所有实体
    entity_type = ['assist', 'cellno', 'city', 'community', 'devzone', 'distance', 'district', 'floorno',
                   'houseno', 'intersection', 'poi', 'prov', 'road', 'roadno', 'subpoi', 'town', 'village_group']

    entity_words = defaultdict(set)
    for line in sentences:
        item = convert_data_format(line)
        for et, words in item['label'].items():
            if et in entity_type:
                for w in words:
                    entity_words[et].add(w)

    new_data.extend(sentences)
    for i in range(num - 1):
        for line in sentences:
            # 随机将同一类型实体进行替换
            item = convert_data_format(line)
            label = item['label']
            if not label:
                new_data.append(line)
                continue
            keys = [k for k in label.keys() if k in entity_type]
            if keys:
                key = random.choice(keys)
                n, spans = random.choice(list(label[key].items()))
                s, e = spans[0]
                new_word = random.choice(list(entity_words[key]))
                ntags = ['I-'+key] * len(new_word)
                ntags[0] = 'B-'+key
                text = list(item['text'])
                tags = [x[1] for x in line]
                new_text = text[:s] + list(new_word) + text[e+1:]
                new_tags = tags[:s] + ntags + tags[e+1:]
                line = [(a, b) for a, b in zip(new_text, new_tags)]

            new_data.append(line)
    return new_data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class NERProcessor(DataProcessor):
    def __init__(self, fold_id=0, fold_num=0):
        self.fold_id = fold_id
        self.fold_num = fold_num

    def get_train_examples(self, data_dir, file_name='train.conll'):
        examples = []
        if FLAGS.extra_pretrain:
            file_name = 'extra_train_v2.conll'

        if self.fold_num > 1:
            sentences = read_data(
                [os.path.join(data_dir, 'train.conll'), os.path.join(data_dir, 'dev.conll')])
            sentences = [line for i,line in enumerate(sentences) if i % self.fold_num != self.fold_id]
        else:
            sentences = read_data(os.path.join(data_dir, file_name))

        if FLAGS.enhance_data:
            sentences = data_enhance(sentences,FLAGS.enhance_num)
        
        for i, line in enumerate(sentences):
            item = convert_data_format(line)
            guid = "%s-%s" % ('train', i)
            text = item['text']
            label = item['label']
            label = self.check(text, label)
            examples.append(InputExample(guid=guid, text=text, label=label))

        # fake 
        if FLAGS.fake_data:
            tf.compat.v1.logging.info("### Using Fake Data ###")
            sentences = read_data(os.path.join(data_dir, 'fake.conll'))
            for i, line in enumerate(sentences):
                if self.fold_num > 0 and i % self.fold_num == self.fold_id:
                    continue
                item = convert_data_format(line)
                guid = "%s-%s" % ('train-fake', i)
                text = item['text']
                label = item['label']
                label = self.check(text, label)
                examples.append(InputExample(guid=guid, text=text, label=label))

        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir, file_name="dev.conll"):
        examples = []
        if self.fold_num > 1:
            sentences = read_data(
                [os.path.join(data_dir, 'train.conll'), os.path.join(data_dir, 'dev.conll')])
        else:
            sentences = read_data(os.path.join(data_dir, file_name))
        for i, line in enumerate(sentences):
            if self.fold_num > 1 and i % self.fold_num != self.fold_id:
                continue
            item = convert_data_format(line)
            guid = '%s-%s' % ('dev', i)
            tags = [x[1] for x in line]
            tags = iobes_iob(tags)
            examples.append(InputExample(
                guid=guid, text=item['text'], label=tags))
        return examples

    def get_test_examples(self, data_dir, file_name="final_test.txt"):
        examples = []
        fname = os.path.join(data_dir, file_name)
        lines = open(fname).readlines()
        if FLAGS.fake_data:
            fname = '../tcdata/final_test.txt'
            lines.extend(open(fname).readlines())
            
        for i, line in enumerate(lines):
            idx, text = line.strip().split('\x01')
            guid = '%s' % (idx)
            examples.append(InputExample(guid=guid, text=text, label=None))
        return examples

    def get_labels(self):
        labels = ['O', 'assist', 'cellno', 'city', 'community', 'devzone', 'distance', 'district', 'floorno',
                  'houseno', 'intersection', 'poi', 'prov', 'road', 'roadno', 'subpoi', 'town', 'village_group']
        return labels

    def check(self, text, label):
        new_labels = []
        for key in label:
            for name, positions in label[key].items():
                for s, e in positions:
                    try:
                        assert text[s:e+1] == name
                    except:
                        # 你不应该来到这里，来了说明数据出问题了
                        pdb.set_trace()
                    new_labels.append((s, e, key))
        return new_labels


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, is_training):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = []
    text = example.text
    if tokenizer.basic_tokenizer.do_lower_case:
        text = text.lower()

    # 及其简化的tokenizer，把每个字符都拆开
    tokens = [
        t if t in tokenizer.vocab else tokenizer.wordpiece_tokenizer.unk_token for t in text]
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        text = text[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    span_mask = []

    ntokens.append("[CLS]")
    segment_ids.append(0)
    span_mask.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        span_mask.append(1)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    span_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        span_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(span_mask) == max_seq_length

    gold_labels = []
    if is_training:
        try:
            ner = {(s, e): label_map[t] for s, e, t in example.label}
        except:
            pdb.set_trace()
        for s in range(len(text)):
            for e in range(s, len(text)):
                gold_labels.append(ner.get((s, e), 0))
    else:
        gold_labels.append(0)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        span_mask=span_mask,
        gold_labels=gold_labels,
    )
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, is_training=True):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.compat.v1.logging.info(
                "Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                         is_training)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features['span_mask'] = create_int_feature(feature.span_mask)
        features['gold_labels'] = create_int_feature(feature.gold_labels)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, batch_size, seq_length, is_training, drop_remainder=False, hvd=None):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "span_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "gold_labels": tf.io.VarLenFeature(tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if name == 'gold_labels':
                t = tf.sparse_tensor_to_dense(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            if hvd is not None:
                d = d.shard(hvd.size(), hvd.rank())
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))

        d = d.padded_batch(
            batch_size,
            padded_shapes={
                "input_ids": (tf.TensorShape([seq_length])),
                "input_mask": tf.TensorShape([seq_length]),
                "segment_ids": tf.TensorShape([seq_length]),
                "span_mask": tf.TensorShape([seq_length]),
                "gold_labels": tf.TensorShape([None])
            },
            padding_values={
                'input_ids': 0,
                "input_mask": 0,
                "segment_ids": 0,
                'span_mask': 0,
                'gold_labels': -1
            },
            drop_remainder=drop_remainder
        )
        return d

    return input_fn


def biaffine_mapping(vector_set_1,
                     vector_set_2,
                     output_size,
                     add_bias_1=True,
                     add_bias_2=True,
                     initializer=None,
                     name='Bilinear'):
    """Bilinear mapping: maps two vector spaces to a third vector space.
    The input vector spaces are two 3d matrices: batch size x bucket size x values
    A typical application of the function is to compute a square matrix
    representing a dependency tree. The output is for each bucket a square
    matrix of the form [bucket size, output size, bucket size]. If the output size
    is set to 1 then results is [bucket size, 1, bucket size] equivalent to
    a square matrix where the bucket for instance represent the tokens on
    the x-axis and y-axis. In this way represent the adjacency matrix of a
    dependency graph (see https://arxiv.org/abs/1611.01734).
    Args:
       vector_set_1: vectors of space one
       vector_set_2: vectors of space two
       output_size: number of output labels (e.g. edge labels)
       add_bias_1: Whether to add a bias for input one
       add_bias_2: Whether to add a bias for input two
       initializer: Initializer for the bilinear weight map
    Returns:
      Output vector space as 4d matrix:
      batch size x bucket size x output size x bucket size
      The output could represent an unlabeled dependency tree when
      the output size is 1 or a labeled tree otherwise.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Dynamic shape info
        batch_size = tf.shape(vector_set_1)[0]
        bucket_size = tf.shape(vector_set_1)[1]

        if add_bias_1:
            vector_set_1 = tf.concat(
                [vector_set_1, tf.ones([batch_size, bucket_size, 1])], axis=2)
        if add_bias_2:
            vector_set_2 = tf.concat(
                [vector_set_2, tf.ones([batch_size, bucket_size, 1])], axis=2)

        # Static shape info
        vector_set_1_size = vector_set_1.get_shape().as_list()[-1]
        vector_set_2_size = vector_set_2.get_shape().as_list()[-1]

        if not initializer:
            initializer = tf.orthogonal_initializer()

        # Mapping matrix
        bilinear_map = tf.get_variable(
            'bilinear_map', [vector_set_1_size,
                             output_size, vector_set_2_size],
            initializer=initializer)

        # The matrix operations and reshapings for bilinear mapping.
        # b: batch size (batch of buckets)
        # v1, v2: values (size of vectors)
        # n: tokens (size of bucket)
        # r: labels (output size), e.g. 1 if unlabeled or number of edge labels.

        # [b, n, v1] -> [b*n, v1]
        vector_set_1 = tf.reshape(vector_set_1, [-1, vector_set_1_size])

        # [v1, r, v2] -> [v1, r*v2]
        bilinear_map = tf.reshape(bilinear_map, [vector_set_1_size, -1])

        # [b*n, v1] x [v1, r*v2] -> [b*n, r*v2]
        bilinear_mapping = tf.matmul(vector_set_1, bilinear_map)

        # [b*n, r*v2] -> [b, n*r, v2]
        bilinear_mapping = tf.reshape(
            bilinear_mapping,
            [batch_size, bucket_size * output_size, vector_set_2_size])

        # [b, n*r, v2] x [b, n, v2]T -> [b, n*r, n]
        bilinear_mapping = tf.matmul(
            bilinear_mapping, vector_set_2, adjoint_b=True)

        # [b, n*r, n] -> [b, n, r, n]
        bilinear_mapping = tf.reshape(
            bilinear_mapping, [batch_size, bucket_size, output_size, bucket_size])
        return bilinear_mapping


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, span_mask, num_labels, use_fgm=False, 
                 perturbation=None, spatial_dropout=None,embedding_dropout=0.0,
                 bilstm=None,biaffine_size=150,pooling_type='last'):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        use_fgm=use_fgm,
        perturbation=perturbation,
        spatial_dropout=spatial_dropout,
        embedding_dropout=embedding_dropout
    )

    output_layer = model.get_sequence_output()

    if pooling_type != 'last':
        raise NotImplementedError('没实现。')

    batch_size, seq_length, hidden_size = modeling.get_shape_list(
        output_layer, expected_rank=3)

    if bilstm is not None and len(bilstm) == 2:
        tf.logging.info('Using Bi-LSTM')
        sequence_length = tf.reduce_sum(input_mask, axis=-1)
        with tf.variable_scope('bilstm', reuse=tf.AUTO_REUSE):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=bilstm[0],
                cell_bw=bilstm[1],
                dtype=tf.float32,
                sequence_length=sequence_length,
                inputs=output_layer
            )
            output_layer = tf.concat(outputs, -1)

    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    # Magic Number
    size = biaffine_size
    
    starts = tf.layers.dense(output_layer, size, kernel_initializer=tf.truncated_normal_initializer(
        stddev=0.02), name='start', reuse=tf.AUTO_REUSE)
    ends = tf.layers.dense(output_layer, size, kernel_initializer=tf.truncated_normal_initializer(
        stddev=0.02), name='end', reuse=tf.AUTO_REUSE)

    biaffine = biaffine_mapping(
        starts,
        ends,
        num_labels,
        add_bias_1=True,
        add_bias_2=True,
        initializer=tf.zeros_initializer())

    # B,L,L,N
    candidate_ner_scores = tf.transpose(biaffine, [0, 1, 3, 2])

    # [B,1,L] [B,L,1] -> [B,L,L]
    span_mask = tf.cast(span_mask, dtype=tf.bool)
    candidate_scores_mask = tf.logical_and(tf.expand_dims(
        span_mask, axis=1), tf.expand_dims(span_mask, axis=2))
    # B,L,L
    sentence_ends_leq_starts = tf.tile(
        tf.expand_dims(
            tf.logical_not(tf.sequence_mask(tf.range(seq_length), seq_length)),
            0),
        [batch_size, 1, 1]
    )
    # B,L,L
    candidate_scores_mask = tf.logical_and(
        candidate_scores_mask, sentence_ends_leq_starts)
    # B*L*L
    flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask, [-1])

    candidate_ner_scores = tf.boolean_mask(tf.reshape(
        candidate_ner_scores, [-1, num_labels]), flattened_candidate_scores_mask)
    return candidate_ner_scores, model


def focal_loss(logits, labels, gamma=2.0):
    epsilon = 1.e-9
    y_pred = tf.nn.softmax(logits, dim=-1)
    y_pred = y_pred + epsilon  # to avoid 0.0 in log
    loss = -labels*tf.pow((1-y_pred), gamma)*tf.log(y_pred)
    return loss


def model_fn_builder(bert_config, num_labels, init_checkpoint=None, learning_rate=None,
                     num_train_steps=None, num_warmup_steps=None,
                     use_one_hot_embeddings=False, hvd=None, amp=False):
    def model_fn(features, labels, mode, params):
        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        span_mask = features["span_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if is_training and FLAGS.bert_dropout > 0.0:
            bert_config.hidden_dropout_prob = FLAGS.bert_dropout
            bert_config.attention_probs_dropout_prob = FLAGS.bert_dropout

        batch_size = tf.shape(input_ids)[0]
        spatial_dropout_layer = None
        if is_training and FLAGS.spatial_dropout > 0.0:
            spatial_dropout_layer = tf.keras.layers.SpatialDropout1D(
                FLAGS.spatial_dropout)

        bilstm = None
        if FLAGS.use_bilstm:
            fw_cell = tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size)
            bw_cell = tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size)
            if is_training:
                fw_cell = lstm_dropout_warpper(fw_cell)
                bw_cell = lstm_dropout_warpper(bw_cell)
            bilstm = (fw_cell, bw_cell)

        reuse_model = FLAGS.use_fgm
        candidate_ner_scores, model = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, span_mask, num_labels, 
            spatial_dropout=spatial_dropout_layer, bilstm=bilstm, use_fgm=reuse_model,
            biaffine_size=FLAGS.biaffine_size,pooling_type=FLAGS.pooling_type,embedding_dropout=FLAGS.embedding_dropout
            )

        output_spec = None
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if init_checkpoint and (hvd is None or hvd.rank() == 0):
                (assignment_map,
                 initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, convert_electra=FLAGS.electra)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.compat.v1.logging.info("**** Trainable Variables ****")

            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                          init_string)

            gold_labels = features['gold_labels']
            gold_labels = tf.boolean_mask(
                gold_labels, tf.not_equal(gold_labels, -1))
            entity_gold_labels = tf.cast(tf.greater(gold_labels,0),candidate_ner_scores.dtype)

            # 真实实体
            true_labels = tf.boolean_mask(
                gold_labels, tf.not_equal(gold_labels, 0))
            pred_labels = tf.boolean_mask(
                candidate_ner_scores, tf.not_equal(gold_labels, 0))
            # 只统计真实实体的准确率，否则准确率虚高
            accuracy = tf.metrics.accuracy(
                true_labels, tf.arg_max(pred_labels, dimension=-1))

            negative_labels = tf.boolean_mask(
                gold_labels, tf.equal(gold_labels, 0))
            negative_pred_labels = tf.boolean_mask(
                candidate_ner_scores, tf.equal(gold_labels, 0))
            # 只统计真实实体的准确率，否则准确率虚高
            negative_accuracy = tf.metrics.accuracy(
                negative_labels, tf.arg_max(negative_pred_labels, dimension=-1))
            tensor_to_log = {
                "positive_accuracy": accuracy[1] * 100,
                "negative_accuracy": negative_accuracy[1] * 100
            }
            if FLAGS.focal_loss:
                gold_labels = tf.one_hot(
                    gold_labels, depth=num_labels, dtype=tf.float32)
                total_loss = focal_loss(candidate_ner_scores, gold_labels)
            else:
                total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=gold_labels, logits=candidate_ner_scores)

            if 0.0 < FLAGS.neg_sample < 1.0:
                # 对负样本进行采样
                sample_vals = tf.random.uniform(shape=tf.shape(gold_labels))
                masks = tf.where_v2(tf.logical_and(
                    gold_labels <= 0, sample_vals >= FLAGS.neg_sample), 0.0, 1.0)
                total_loss = masks * total_loss

            total_loss = tf.reduce_sum(total_loss) / tf.to_float(batch_size)

            
            if FLAGS.use_fgm:
                embedding_output = model.get_embedding_output()
                grad, = tf.gradients(
                    total_loss,
                    embedding_output,
                    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                grad = tf.stop_gradient(grad)
                perturbation = modeling.scale_l2(grad, FLAGS.fgm_epsilon)
                adv_candidate_ner_scores, _ = create_model(
                    bert_config, is_training, input_ids, input_mask, segment_ids, span_mask, num_labels,
                    use_fgm=True, perturbation=perturbation, spatial_dropout=spatial_dropout_layer, bilstm=bilstm,
                    biaffine_size=FLAGS.biaffine_size,pooling_type=FLAGS.pooling_type,embedding_dropout=FLAGS.embedding_dropout
                    )

                if FLAGS.focal_loss:
                    adv_loss = focal_loss(
                        adv_candidate_ner_scores, gold_labels)
                else:
                    adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=gold_labels, logits=adv_candidate_ner_scores)

                # 对adv_loss进行sample会导致效果下降
                # if 0.0 < FLAGS.neg_sample < 1.0:
                #     adv_loss = masks * adv_loss
                adv_loss = tf.reduce_sum(adv_loss) / tf.to_float(batch_size)


                total_loss = (total_loss + FLAGS.fgm_loss_ratio *
                              adv_loss) / (1 + FLAGS.fgm_loss_ratio)

            train_op, _ = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd, amp, head_lr_ratio=FLAGS.head_lr_ratio)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[tf.train.LoggingTensorHook(tensor_to_log, every_n_iter=50)])
        elif mode == tf.estimator.ModeKeys.EVAL:
            # Fake metric
            def metric_fn():
                unused_mean = tf.metrics.mean(tf.ones([2, 3]))
                return {
                    "unused_mean": unused_mean
                }
            eval_metric_ops = metric_fn()
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.constant(1.0),
                eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"score": tf.expand_dims(candidate_ner_scores, 0), 'batch_size': tf.expand_dims(
                    batch_size, 0), "prob":tf.expand_dims(tf.nn.softmax(candidate_ner_scores),0)} 
            )
        return output_spec

    return model_fn


def main(_):
    # Set different seed for different model
    seed = FLAGS.seed + FLAGS.fold_id
    tf.random.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.horovod:
        import horovod.tensorflow as hvd
        hvd.init()

    processors = {
        "ner": NERProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_train_and_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    processor = processors[task_name](FLAGS.fold_id,FLAGS.fold_num)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    master_process = True
    training_hooks = []
    global_batch_size = FLAGS.train_batch_size
    hvd_rank = 0

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if FLAGS.horovod:
        global_batch_size = FLAGS.train_batch_size * hvd.size()
        master_process = (hvd.rank() == 0)
        hvd_rank = hvd.rank()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        if hvd.size() > 1:
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        if FLAGS.amp:
            tf.enable_resource_variables()

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir if master_process else None,
        session_config=config,
        log_step_count_steps=50,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
        keep_checkpoint_max=1)

    if master_process:
        tf.compat.v1.logging.info("***** Configuaration *****")
        for key in FLAGS.__flags.keys():
            tf.compat.v1.logging.info(
                '  {}: {}'.format(key, getattr(FLAGS, key)))
        tf.compat.v1.logging.info("**************************")

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train or FLAGS.do_train_and_eval:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / global_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        start_index = 0
        end_index = len(train_examples)
        tmp_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]

        if FLAGS.horovod:
            tmp_filenames = [os.path.join(
                FLAGS.output_dir, "train.tf_record{}".format(i)) for i in range(hvd.size())]
            num_examples_per_rank = len(train_examples) // hvd.size()
            remainder = len(train_examples) % hvd.size()
            if hvd.rank() < remainder:
                start_index = hvd.rank() * (num_examples_per_rank+1)
                end_index = start_index + num_examples_per_rank + 1
            else:
                start_index = hvd.rank() * num_examples_per_rank + remainder
                end_index = start_index + (num_examples_per_rank)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate if not FLAGS.horovod else FLAGS.learning_rate * hvd.size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        hvd=None if not FLAGS.horovod else hvd,
        amp=FLAGS.amp)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if FLAGS.do_train or FLAGS.do_train_and_eval:
        filed_based_convert_examples_to_features(
            train_examples[start_index:end_index], label_list, FLAGS.max_seq_length, tokenizer, tmp_filenames[hvd_rank], True)
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=tmp_filenames,  # train_file,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            hvd=None if not FLAGS.horovod else hvd)

    if FLAGS.do_predict or FLAGS.do_eval or FLAGS.do_train_and_eval:
        if FLAGS.do_eval or FLAGS.do_train_and_eval:
            predict_examples = processor.get_dev_examples(FLAGS.data_dir)
        else:
            predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, 'predict.tf_record')
        filed_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, False)
        predict_batch_size = FLAGS.predict_batch_size
        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d", len(predict_examples))
        tf.compat.v1.logging.info("  Batch size = %d", predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            batch_size=predict_batch_size,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False
        )

    if (FLAGS.do_train_and_eval or FLAGS.do_train) and FLAGS.start_swa_step > 0:
        checkpoint_path = FLAGS.output_dir + '/swa'
        tf.io.gfile.makedirs(checkpoint_path)
        swa_hook = SWAHook(
            FLAGS.swa_steps, FLAGS.start_swa_step, checkpoint_path)
        training_hooks.append(swa_hook)

    if FLAGS.do_train_and_eval:
        exporter = BestF1Exporter(
            predict_input_fn, predict_examples, label_list, FLAGS.max_seq_length, dp=FLAGS.dp_decode)
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=num_train_steps, hooks=training_hooks)
        # 我们不想跑这么多步，毕竟不是用eval的metric来保存模型
        eval_spec = tf.estimator.EvalSpec(
            input_fn=predict_input_fn, steps=2, exporters=exporter, start_delay_secs=0, throttle_secs=0)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        if FLAGS.do_train:
            estimator.train(input_fn=train_input_fn,
                            max_steps=num_train_steps, hooks=training_hooks)

    if FLAGS.do_eval or FLAGS.do_predict:
        final_results = []
        idx = 0
        for i, prediction in enumerate(tqdm(estimator.predict(input_fn=predict_input_fn, yield_single_examples=True), total=len(predict_examples)//predict_batch_size)):
            scores = prediction['score']
            if FLAGS.export_prob:
                scores = prediction['prob']
            offset = 0
            bz = prediction['batch_size']
            for j in range(bz):
                example = predict_examples[idx]
                text = example.text
                pred_text = example.text[:FLAGS.max_seq_length-2]
                size = len(pred_text) * (len(pred_text) + 1) // 2
                pred_score = scores[offset:offset+size]
                idx += 1
                offset += size
                if FLAGS.export_prob:
                    results = get_biaffine_pred_prob(pred_text,pred_score,label_list)
                    final_results.append({
                        "id":example.guid,
                        'text':text,
                        'prob':results
                    })
                else:
                    if FLAGS.dp_decode:
                        results = get_biaffine_pred_ner_with_dp(pred_text,pred_score)
                    else:
                        results = get_biaffine_pred_ner(pred_text, pred_score)
                    labels = {}
                    for s, e, t, score in results:
                        span = text[s:e+1]
                        label = label_list[t]
                        item = [s, e]
                        if label not in labels:
                            labels[label] = {span: [item]}
                        else:
                            if span in labels[label]:
                                labels[label][span].append(item)
                            else:
                                labels[label][span] = [item]
                    tags = convert_back_to_bio(labels, text)
                    if FLAGS.do_eval:
                        tags = [' '.join([c, t, p])
                                for c, t, p in zip(text, example.label, tags)]
                    else:
                        tags = iob_iobes(tags)
                        tags = '\x01'.join([example.guid, text, ' '.join(tags)])
                    final_results.append(tags)

        assert len(final_results) == len(predict_examples)
        if FLAGS.do_eval:
            eval_lines, f1 = eval_ner(final_results, FLAGS.output_dir, 'eval')
            for line in eval_lines:
                print(line.rstrip())
            print('f1-{}'.format(f1))
            with open(os.path.join(FLAGS.output_dir, 'eval_f1.txt'), 'w') as f:
                f.write("best f1: {}".format(f1))
        else:
            with open(os.path.join(FLAGS.output_dir, 'result.txt'), 'w') as f:
                for x in final_results:
                    if FLAGS.export_prob:
                        f.write(json.dumps(x,ensure_ascii=False) + '\n')
                    else:
                        f.write(x+'\n')

    end_time = time.time()
    with open('/tmp/time.txt','a') as f:
        if FLAGS.do_predict:
            f.write('预测{}用时{}\n'.format(FLAGS.output_dir,end_time-start_time))
        else:
            f.write('训练{}用时{}\n'.format(FLAGS.output_dir,end_time-start_time))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
