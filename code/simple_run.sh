#!/usr/bin/env bash

# 数据预处理
mkdir -p ../user_data/tcdata
mkdir -p ../user_data/texts

cp -r ../tcdata ../user_data

echo "Data preprocess"
python create_raw_text.py

# 预训练

cd electra-pretrain

export DATA_DIR=../../user_data
export ELECTRA_DIR=../../user_data/electra

echo 'Prepare pretraining data...'
python build_pretraining_dataset.py \
  --corpus-dir=${DATA_DIR}/texts \
  --max-seq-length=32 \
  --vocab-file=${ELECTRA_DIR}/electra_180g_base/vocab.txt \
  --output-dir=${DATA_DIR}/pretrain_tfrecords 

echo "Pretrain base electra model ~= 1 hour on Titan RTX 24G"
# 如果只有12G显存，可以将train_batch_size改成32，accumulation_step设置为3，耗时久一点，效果应该差不多
python run_pretraining.py \
  --data-dir=${DATA_DIR} \
  --model-name=base \
  --hparams='{"max_seq_length":32, "accumulation_step": 2, "use_amp": true, "learning_rate": 0.0002,"model_size": "base","eval_batch_size":128,"train_batch_size": 64, "init_checkpoint": "../../user_data/electra/electra_180g_base/electra_180g_base.ckpt", "vocab_file": "../../user_data/electra/electra_180g_base/vocab.txt"}'

cd ..

# 利用额外数据训练

export BERT_DIR=../user_data/electra/electra_180g_base
export CONFIG_FILE=../user_data/electra/electra_180g_base/base_discriminator_config.json
export INIT_CHECKPOINT=../user_data/models/base/model.ckpt-7000
export DATA_DIR=../user_data/tcdata
export SEED=20190525
export EMBEDDING_DROPOUT=0.1
export OUTPUT_DIR=../user_data/models/bif_extra_enhance_electra_base_pretrain

python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file=${BERT_DIR}/vocab.txt \
  --bert_config_file=${CONFIG_FILE} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --do_lower_case=True \
  --max_seq_length=32 \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --neg_sample=1.0 \
  --save_checkpoints_steps=450 \
  --do_train_and_eval=true \
  --do_train=false \
  --do_eval=false \
  --do_predict=false \
  --use_fgm=true \
  --fgm_epsilon=0.8 \
  --fgm_loss_ratio=1.0 \
  --spatial_dropout=0.3 \
  --embedding_dropout=${EMBEDDING_DROPOUT} \
  --head_lr_ratio=1.0 \
  --pooling_type=last \
  --extra_pretrain=true \
  --enhance_data=true \
  --electra=true \
  --dp_decode=true \
  --amp=false \
  --seed=${SEED} \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}


# finetune

export INIT_CHECKPOINT=../user_data/models/bif_extra_enhance_electra_base_pretrain/export/f1_export/model.ckpt
export OUTPUT_DIR=../user_data/models/k-fold/bif_electra_base_pretrain
export SEED=666
export SPATIAL_DROPOUT=0.1
export EMBEDDING_DROPOUT=0.1

python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file=${BERT_DIR}/vocab.txt \
  --bert_config_file=${CONFIG_FILE} \
  --init_checkpoint=${INIT_CHECKPOINT}  \
  --do_lower_case=True \
  --max_seq_length=32 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --neg_sample=1.0 \
  --save_checkpoints_steps=276 \
  --do_train_and_eval=true \
  --do_train=false \
  --do_eval=false \
  --do_predict=false \
  --use_fgm=true \
  --fgm_epsilon=0.8 \
  --fgm_loss_ratio=1.0 \
  --spatial_dropout=${SPATIAL_DROPOUT} \
  --embedding_dropout=${EMBEDDING_DROPOUT} \
  --head_lr_ratio=1.0 \
  --pooling_type=last \
  --start_swa_step=0 \
  --swa_steps=100 \
  --biaffine_size=150 \
  --electra=false \
  --dp_decode=true \
  --amp=false \
  --seed=${SEED} \
  --data_dir=${DATA_DIR} \
  --output_dir=${OUTPUT_DIR}