import subprocess
import time
import os
import logging
import copy

import threading
from multiprocessing import Process
import multiprocessing as mp

logging.basicConfig()
logger = logging.getLogger('pipeline')
fh = logging.FileHandler('/tmp/pipeline.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600

def train_model(args,cmd):
    logger.info('start to train model {}: {}'.format(args.get('OUTPUT_DIR',''),timer.get_current_time()))
    os.system(cmd.format(**args))
    logger.info('finish train model {}: {}'.format(args.get('OUTPUT_DIR',''),timer.get_current_time()))

timer = Timer()

# 数据准备
logger.info(f'data prepare start: {timer.get_current_time()} ')

prepare = subprocess.Popen(
    "bash prepare.sh", shell=True
)
prepare.wait()
logger.info(f'data prepare finished: {timer.get_current_time()}')

# 预训练
logger.info(f'electra pretrain start: {timer.get_current_time()} ')

pretrain = subprocess.Popen(
    "bash pretrain.sh", shell=True
)
pretrain.wait()

logger.info(f'electra pretrain finished: {timer.get_current_time()}')


# extra pretrain
basic_base_args = {
  "BERT_DIR":"../user_data/electra/electra_180g_base",
  "CONFIG_FILE":"../user_data/electra/electra_180g_base/base_discriminator_config.json",
  "INIT_CHECKPOINT":"../user_data/models/base/model.ckpt-7000",
  "DATA_DIR":'../user_data/tcdata',
  "SEED":20190525,
  "EMBEDDING_DROPOUT":0.1,
  "OUTPUT_DIR":"../user_data/models/bif_extra_enhance_electra_base_pretrain"
}

basic_large_args = {
  "BERT_DIR":"../user_data/electra/electra_180g_large",
  "CONFIG_FILE":"../user_data/electra/electra_180g_large/large_discriminator_config.json",
  "INIT_CHECKPOINT":"../user_data/models/large/model.ckpt-5000",
  "DATA_DIR":'../user_data/tcdata',
  "SEED":807,
  "EMBEDDING_DROPOUT":0.2,
  "OUTPUT_DIR":"../user_data/models/bif_extra_enhance_electra_large_pretrain"
}

bif_extra_cmd = '''
python run_biaffine_ner.py \
  --task_name=ner \
	--vocab_file={BERT_DIR}/vocab.txt \
  --bert_config_file={CONFIG_FILE} \
  --init_checkpoint={INIT_CHECKPOINT} \
  --do_lower_case=True \
  --max_seq_length=64 \
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
  --embedding_dropout={EMBEDDING_DROPOUT} \
  --head_lr_ratio=1.0 \
  --pooling_type=last \
  --extra_pretrain=true \
  --enhance_data=true \
  --electra=true \
  --dp_decode=true \
  --amp=true \
  --seed={SEED} \
  --data_dir={DATA_DIR} \
  --output_dir={OUTPUT_DIR}
'''

pool = mp.Pool(processes = 2)

# base bif
pool.apply_async(train_model,(basic_base_args, bif_extra_cmd))

# large bif
pool.apply_async(train_model,(basic_large_args, bif_extra_cmd))


pool.close()
pool.join()


# finetune
bif_finetune_cmd = '''
python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file={BERT_DIR}/vocab.txt \
  --bert_config_file={CONFIG_FILE} \
  --init_checkpoint={INIT_CHECKPOINT}  \
  --do_lower_case=True \
  --max_seq_length=64 \
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
  --spatial_dropout={SPATIAL_DROPOUT} \
  --embedding_dropout={EMBEDDING_DROPOUT} \
  --head_lr_ratio=1.0 \
  --pooling_type=last \
  --start_swa_step=0 \
  --swa_steps=100 \
  --biaffine_size=150 \
  --electra=false \
  --dp_decode=true \
  --amp=true \
  --seed={SEED} \
  --fold_id={FOLD_ID} \
  --fold_num={FOLD_NUM} \
  --data_dir={DATA_DIR} \
  --output_dir={OUTPUT_DIR}
'''

basic_base_finetune_args = {
  "BERT_DIR":"../user_data/electra/electra_180g_base",
  "CONFIG_FILE":"../user_data/electra/electra_180g_base/base_discriminator_config.json",
  "INIT_CHECKPOINT":"../user_data/models/bif_extra_enhance_electra_base_pretrain/export/f1_export/model.ckpt",
  "DATA_DIR":'../user_data/tcdata',
  "OUTPUT_DIR":'',
  "SEED": 666,
  "SPATIAL_DROPOUT":0.1,
  "EMBEDDING_DROPOUT":0.1,
  "FOLD_ID": 0,
  "FOLD_NUM": 5
}

basic_large_finetune_args = {
  "BERT_DIR":"../user_data/electra/electra_180g_large",
  "CONFIG_FILE":"../user_data/electra/electra_180g_large/large_discriminator_config.json",
  "INIT_CHECKPOINT":"../user_data/models/bif_extra_enhance_electra_large_pretrain/export/f1_export/model.ckpt",
  "DATA_DIR":'../user_data/tcdata',
  "OUTPUT_DIR":'',
  "SEED": 777,
  "SPATIAL_DROPOUT":0.2,
  "EMBEDDING_DROPOUT":0.2,
  "FOLD_ID": 0,
  "FOLD_NUM": 5
}


base_outdir_format = "../user_data/models/k-fold/bif_electra_base_pretrain_fold_{}"
large_outdir_format = "../user_data/models/k-fold/bif_electra_large_pretrain_fold_{}"

pool = mp.Pool(processes = 3)
for i in range(5):
  # base
  # bif
  args = copy.deepcopy(basic_base_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = base_outdir_format.format(i)
  pool.apply_async(train_model,(args, bif_finetune_cmd))

pool.close()
pool.join()

pool = mp.Pool(processes = 1)
for i in range(5):
  args = copy.deepcopy(basic_large_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = large_outdir_format.format(i)
  pool.apply_async(train_model,(args, bif_finetune_cmd))

pool.close()
pool.join()

bif_pred_cmd = '''
python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file={BERT_DIR}/vocab.txt \
  --bert_config_file={CONFIG_FILE} \
  --do_lower_case=True \
  --max_seq_length=64 \
  --do_predict=true \
  --use_fgm=true \
  --pooling_type=last \
  --biaffine_size=150 \
  --dp_decode=true \
  --fake_data=true \
  --fold_id={FOLD_ID} \
  --fold_num={FOLD_NUM} \
  --data_dir={DATA_DIR} \
  --output_dir={OUTPUT_DIR}/export/f1_export
'''

pool = mp.Pool(processes = 4)
for i in range(5):
  # # base
  args = copy.deepcopy(basic_base_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = base_outdir_format.format(i)
  pool.apply_async(train_model,(args, bif_pred_cmd))

  args = copy.deepcopy(basic_large_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = large_outdir_format.format(i)
  pool.apply_async(train_model,(args, bif_pred_cmd))

pool.close()
pool.join()

logger.info(f'assemble fake start: {timer.get_current_time()}')
from assemble import assemble_fake
assemble_fake()
logger.info(f'assemble fake finish: {timer.get_current_time()}')

bif_fake_cmd = '''
python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file={BERT_DIR}/vocab.txt \
  --bert_config_file={CONFIG_FILE} \
  --init_checkpoint={INIT_CHECKPOINT}  \
  --do_lower_case=True \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --neg_sample=0.15 \
  --save_checkpoints_steps=500 \
  --do_train_and_eval=true \
  --do_train=false \
  --do_eval=false \
  --do_predict=false \
  --use_fgm=true \
  --fgm_epsilon=0.8 \
  --fgm_loss_ratio=1.0 \
  --spatial_dropout=0.1 \
  --embedding_dropout=0.1 \
  --head_lr_ratio=1.0 \
  --pooling_type=last \
  --start_swa_step=0 \
  --swa_steps=100 \
  --biaffine_size=150 \
  --electra=false \
  --dp_decode=true \
  --amp=true \
  --fake_data=true \
  --fold_id={FOLD_ID} \
  --fold_num={FOLD_NUM} \
  --data_dir={DATA_DIR} \
  --output_dir={OUTPUT_DIR}
'''

fake_outdir_format = "../user_data/models/k-fold/bif_fake_tags_fold_{}"

pool = mp.Pool(processes = 3)
for i in range(5):
  args = copy.deepcopy(basic_base_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = fake_outdir_format.format(i)
  pool.apply_async(train_model,(args, bif_fake_cmd))

pool.close()
pool.join()

fake_pred_cmd = '''
python run_biaffine_ner.py \
  --task_name=ner \
  --vocab_file={BERT_DIR}/vocab.txt \
  --bert_config_file={CONFIG_FILE} \
  --do_lower_case=True \
  --max_seq_length=64 \
  --do_predict=true \
  --use_fgm=true \
  --pooling_type=last \
  --biaffine_size=150 \
  --dp_decode=true \
  --fake_data=false \
  --fold_id={FOLD_ID} \
  --fold_num={FOLD_NUM} \
  --data_dir={DATA_DIR} \
  --output_dir={OUTPUT_DIR}/export/f1_export
'''

pool = mp.Pool(processes = 5)
for i in range(5):
  args = copy.deepcopy(basic_base_finetune_args)
  args['FOLD_ID'] = i
  args['OUTPUT_DIR'] = fake_outdir_format.format(i)
  pool.apply_async(train_model,(args, fake_pred_cmd))

pool.close()
pool.join()

from assemble import assemble_final
assemble_final()