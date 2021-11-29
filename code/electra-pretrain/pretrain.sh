export DATA_DIR=../../user_data
export ELECTRA_DIR=../../user_data/electra

echo 'Prepare pretraining data...'
python build_pretraining_dataset.py \
  --corpus-dir=${DATA_DIR}/texts \
  --max-seq-length=64 \
  --vocab-file=${ELECTRA_DIR}/electra_180g_base/vocab.txt \
  --output-dir=${DATA_DIR}/pretrain_tfrecords 

echo "Pretrain base electra model ~= 1 hour on V100"
python run_pretraining.py \
  --data-dir=${DATA_DIR} \
  --model-name=base \
  --hparams='{"use_amp": true, "learning_rate": 0.0002,"model_size": "base","eval_batch_size":128,"train_batch_size": 128, "init_checkpoint": "../../user_data/electra/electra_180g_base/electra_180g_base.ckpt", "vocab_file": "../../user_data/electra/electra_180g_base/vocab.txt"}'


echo "pretrain large electra model ~= 2.2 hours on V100"
python run_pretraining.py \
  --data-dir=${DATA_DIR} \
  --model-name=large \
  --hparams='{"num_train_steps": 5000, "num_warmup_steps": 500, "model_size": "large", "train_batch_size": 43, "learning_rate": 5e-05, "init_checkpoint": "../../user_data/electra/electra_180g_large/electra_180g_large.ckpt", "use_amp": true, "accumulation_step": 3, "vocab_file": "../../user_data/electra/electra_180g_large/vocab.txt"}'
