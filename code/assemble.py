'''
模型结果融合
'''
import re
from collections import Counter, defaultdict
from glob import glob

from utils import convert_data_format, iob_iobes


def refine_entity(w,s,e):
  # 去除包含特殊字符的实体
  if re.findall('[，。（）()]',w):
    nw = w.strip('，。（）()')
    if not nw:
      return False,None
    else:
      start = w.find(nw)
      s = s + start
      e = s + len(nw) - 1
      return True,(s,e)
  else:
    return True,(s,e)

def convert(entity, refine=False):
    tmp = []
    for k,words in entity.items():
        for w,spans in words.items():
            for span in spans:
                if refine:
                    should_keep,span = refine_entity(w,span[0],span[1])
                    if not should_keep:
                        continue
                tmp.append((k,w,span[0],span[1]))
    return tmp

def get_entities(text,tags):
    tag_words = []
    word = ''
    tag = ''
    for i,(c,t) in enumerate(zip(text,tags)):
        if t[0] in ['B','S','O']:
            if word:
                tag_words.append((word,i,tag))
            if t[0] == 'O':
                word = ''
                tag = ''
                continue
            word = c
            tag = t[2:]
        else:
            word += c
    if word:
        tag_words.append((word,i+1,tag))

    entities = {}
    for w,i,t in tag_words:
        if t not in entities:
            entities[t] = {}
        if w in entities[t]:
            entities[t][w].append([i-len(w),i-1])
        else:
            entities[t][w] = [[i-len(w),i-1]]
    return entities
  
def check_special(text):
  text = re.sub('[\u4e00-\u9fa5]','',text)
  text = re.sub('[0A-]','',text)
  if text.strip():
    return True
  else:
    return False

def merge_by_4_tuple(raw_texts,data,weights,threshold=3.0, refine=False):
  '''
  根据（类型、实体文本、起始位置、结束位置）四元组进行投票确定最终的结果
  '''
  new_tags = []
  ent_cnt = 0
  special_cnt = 0
  check_fail = 0
  fail_cnt = 0

  for i,gtags in enumerate(data):
    _,text = raw_texts[i]
    cnt = Counter()
    assert len(weights) == len(gtags), 'weight {} != tags {}'.format(len(weights),len(gtags))
    for j,tags in enumerate(gtags):
      entities = convert(get_entities(text,tags))
      ratio = weights[j]
      for x in entities:
        cnt[x] += ratio

    ntags = ['O'] * len(text)
    for m,n in cnt.most_common():
      # k = 类型, w = 实体文本, s = 实体起始位置, e = 实体结束位置
      (k,w,s,e) = m
      if n < threshold:
        fail_cnt += 1
        continue

      if refine:
        should_keep,span = refine_entity(w,s,e)
        if not should_keep:
          continue
        else:
          s,e = span

      # 检查是否有其他实体占据span
      if not all(x=='O' for x in ntags[s:e+1]):
        continue

      ent_cnt += 1
      try:
        if check_special(text[s:e+1]):
          special_cnt += 1
      except:
        check_fail += 1
      ntags[s:e+1] = ['I-'+k] * (e-s+1)
      ntags[s] = 'B-'+k
    new_tags.append(iob_iobes(ntags))

  with open('/tmp/entity_cnt.txt','w') as f:
    f.write('fail_cnt - {}, ent_cnt - {}, special_cnt - {}\n'.format(fail_cnt,ent_cnt,special_cnt))

  return new_tags


def assemble_fake():
  base_dir = '../user_data/models'
  output_file= '../user_data/tcdata/fake.conll'

  patterns = [
    base_dir + '/k-fold/bif_electra_base_pretrain_fold_*/export/f1_export/result.txt',
    base_dir + '/k-fold/bif_electra_large_pretrain_fold_*/export/f1_export/result.txt',
  ]

  weights = [1/2] * 5 + [1/2] * 5 
  threshold = 3.0
  refine = True
  
  data = []
  raw_texts = []

  for pattern in patterns:
    for fname in glob(pattern):
      for i,line in enumerate(open(fname)):
        idx,text,tags = line.strip().split('\x01')
        if len(data) <= i:
          data.append([])
        data[i].append(tags.split(' '))
        if len(raw_texts) <= i:
          raw_texts.append((idx,text))
  
  assert len(data[0]) == len(weights)
  new_tags = merge_by_4_tuple(raw_texts,data,weights,threshold,refine)

  seen_texts = set()

  with open(output_file,'w') as f:
    for (idx,text),tags in zip(raw_texts,new_tags):
      if len(text) != len(tags):
        continue
      if text in seen_texts:
        continue
      else:
        seen_texts.add(text)
        
      for c,t in zip(text,tags):
        f.write(c + ' ' + t + '\n')
      f.write('\n')

def assemble_final():
  base_dir = '../user_data/models'
  output_file= './result.txt'

  patterns = [
    base_dir + '/k-fold/bif_fake_tags_fold_*/export/f1_export/result.txt',
  ]

  weights = [1] * 5 
  threshold = 3.0
  refine = True
  
  data = []
  raw_texts = []

  for pattern in patterns:
    for fname in glob(pattern):
      for i,line in enumerate(open(fname)):
        idx,text,tags = line.strip().split('\x01')
        if len(data) <= i:
          data.append([])
        data[i].append(tags.split(' '))
        if len(raw_texts) <= i:
          raw_texts.append((idx,text))
  
  assert len(data[0]) == len(weights)
  new_tags = merge_by_4_tuple(raw_texts,data,weights,threshold,refine)

  with open(output_file,'w') as f:
    for (idx,text),tags in zip(raw_texts,new_tags):
        assert len(text) == len(tags)
        f.write('\x01'.join([idx,text,' '.join(tags)]) + '\n')