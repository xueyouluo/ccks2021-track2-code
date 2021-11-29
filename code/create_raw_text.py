import re
import json
import random

from collections import Counter,defaultdict

from utils import normalize, read_data, convert_back_to_bio, convert_data_format, iob_iobes

random.seed(20190525)

TCDATA_DIR = '../user_data/tcdata/'
USERDATA_DIR = '../user_data/'


def read_conll(fname):
    lines = []
    line = ''
    for x in open(fname):
        x = x.strip()
        if not x:
            lines.append(line)
            line = ''
            continue
        else:
            line += x.split(' ')[0]
    return lines

def read_track3(fname):
    lines = []
    for x in open(fname):
        x = json.loads(x)
        lines.append(x['query'])
        for y in x['candidate']:
            lines.append(y['text'])
    return [normalize(x) for x in lines]

def create_preatrain_data():
    # 构建预训练语料
    data = open(TCDATA_DIR + 'final_test.txt').readlines()
    data = [x.strip().split('\x01')[1] for x in data]
    train = read_conll(TCDATA_DIR + 'train.conll')
    dev = read_conll(TCDATA_DIR + 'dev.conll')
    # 复赛没有使用
    # train3 = read_track3(
    #     USERDATA_DIR + 'track3/Xeon3NLP_round1_train_20210524.txt')
    # test3 = read_track3(
    #     USERDATA_DIR + 'track3/Xeon3NLP_round1_test_20210524.txt')
    extra_data = read_data([USERDATA_DIR + 'extra_data/train.txt', USERDATA_DIR +
                      'extra_data/dev.txt', USERDATA_DIR + 'extra_data/test.txt'])
    extra_data = [''.join([x[0] for x in item]) for item in extra_data]
    extra_data = [normalize(x) for x in extra_data]
    # old_test = open(USERDATA_DIR + 'track3/final_test.txt').readlines()
    # old_test = [x.strip().split('\x01')[1] for x in old_test]

    texts = list(set(data+train+dev+extra_data))
    texts = [t for t in texts if t.strip()]
    random.shuffle(texts)

    with open(USERDATA_DIR + 'texts/raw_text.txt', 'w') as f:
        for x in texts:
            f.write(x+'\n')

def convert_distance(item,tags):
    # 根据规则将assit中与距离相关的转换为distance标签
    text = item['text']
    spans = [x for x in re.finditer('(0+|(十?[一二三四五六七八九几]+(十|百)?[一二三四五六七八九几]?))米',text)]
    for sp in spans:
        start,end = sp.span()
        if tags[start][2:] == 'assist':
            tags[start:end] = ['I-distance'] * (end-start)
            tags[start] = 'B-distance'
            if end < len(tags) and tags[end][0] == 'I':
                tags[end] = 'B' + tags[end][1:]
    return tags,spans

def convert_village(item,tags):
    # 根据规则转换village_group标签
    text = item['text']
    spans = [x for x in re.finditer('(0+|(十?[一二三四五六七八九])|([一二三四五六七八九]十[一二三四五六七八九]?))[组队社]',text)]
    for sp in spans:
        start,end = sp.span()
        if start > 0 and tags[start-1][2:] == 'community':
            tags[start:end] = ['I-village_group'] * (end-start)
            tags[start] = 'B-village_group'
            if end < len(tags) and tags[end][0] == 'I':
                tags[end] = 'B' + tags[end][1:]
    return tags, spans

def convert_intersection(item,tags,pattern):
    # 根据在训练验证集出现过的intersection字段对标签进行转换
    text = item['text']
    spans = [x for x in re.finditer(pattern,text)]
    for sp in spans:
        start,end = sp.span()
        if tags[start][2:] == 'assist' or text[start:end] == '路口':
            if text[start:end] == '路口':
                if tags[start][2:] == 'road' and tags[start+1][2:] == 'assist':
                    start = start + 1
                elif tags[start-1][2:] == 'road' and text[start-1] not in ['街','路']:
                    tags[start] = 'I-road'
                    start = start + 1
            tags[start:end] = ['I-intersection'] * (end-start)
            tags[start] = 'B-intersection'
            if end < len(tags) and tags[end][0] == 'I':
                tags[end] = 'B' + tags[end][1:]
    return tags,spans

def get_intersection_pattern():
    # 根据赛道2的训练数据获取路口的模式匹配
    train = read_data(TCDATA_DIR+'train.conll')
    dev = read_data(TCDATA_DIR+'dev.conll')
    train = [convert_data_format(x) for x in train]
    dev = [convert_data_format(x) for x in dev]

    inter_cnt = Counter()
    for x in train+dev:
        inter = x['label'].get('intersection','')
        if inter:
            for k in inter:
                inter_cnt[k] += 1

    inter_words = [x[0] for x in inter_cnt.most_common() if len(x[0]) > 1]
    pattern = '|'.join(['({})'.format(x) for x in inter_words])
    return pattern

def check_devzone(name):
    for x in ['经济开发区','园区','开发区','工业园','工业区','科技园','工业园区','创意园','产业园','软件谷','软件园','电商园','智慧国','智慧园','未来科技城','科创中心','机电城','工业城','商务园']:
        if name.endswith(x):
            return True
    return False

def convert_data_format_v2(sentence):
    word = ''
    tag = ''
    text = ''
    tag_words = []
    for i,(c,t) in enumerate(sentence):
        c = normalize(c)
        if t[0] in ['B','S','O']:
            if word:
                tag_words.append((word,len(text),tag))
            if t[0] == 'O':
                word = ''
                tag = ''
                continue
            word = c
            tag = t[2:]
        else:
            word += c
        text += c
        
    if word:
        tag_words.append((word,len(text),tag))
        

    entities = {}
    for w,i,t in tag_words:
        if check_devzone(w):
            t = 'devzone'
        if t not in entities:
            entities[t] = {}
        if w in entities[t]:
            entities[t][w].append([i-len(w),i-1])
        else:
            entities[t][w] = [[i-len(w),i-1]]
    
    return {"text":text,"label":entities}

def _get_refine_entity(raw_files):
    data = read_data(raw_files)
    ent_tp_cnt = defaultdict(Counter)
    ent_cnt = Counter()
    for sentence in data:
        entities = convert_data_format(sentence)['label']
        for k in entities:
            for name in entities[k]:
                ent_tp_cnt[name][k] += 1
                ent_cnt[name] += 1
    
    for name in ent_tp_cnt:
        if ent_cnt[name] < 10:
                continue
        if len(ent_tp_cnt[name]) == 1:
            continue
        if len(ent_tp_cnt[name]) >= 2:
            pop = []
            for tp in ent_tp_cnt[name]:
                if ent_tp_cnt[name][tp] / ent_cnt[name] < 0.1 and ent_tp_cnt[name][tp] < 5:
                    pop.append(tp)
            for tp in pop:
                ent_tp_cnt[name].pop(tp)
    return ent_tp_cnt

def _fix_data(ent_tp_cnt, update_files, iob=False):
    data = read_data(update_files)
    new_data = []
    wcnt = 0
    for sentence in data:
        entities = convert_data_format(sentence)['label']
        new_entities = {}
        for k in entities:
            for name in entities[k]:
                spans = entities[k][name]
                cnt = ent_tp_cnt[name]
                nk = k
                if k not in cnt:
                    # print(''.join([w[0] for w in sentence]))
                    try:
                        nk = ent_tp_cnt[name].most_common(1)[0][0]
                    except:
                        # print('no entity', name,ent_tp_cnt[name],k,entities[k])
                        continue
                    # print("wrong:",name,k,'->',nk)
                    wcnt += 1
                new_entities[nk] = {}
                new_entities[nk][name] = spans
        if iob:
            tags = convert_back_to_bio(new_entities,[w[0] for w in sentence])
        else:
            tags = iob_iobes(convert_back_to_bio(new_entities,[w[0] for w in sentence]))
        new_data.append([(a[0],b) for a,b in zip(sentence,tags)])
    print('# total wrong',wcnt)
    return new_data

def fix_data():
    ent_tp_cnt = _get_refine_entity([TCDATA_DIR + 'train.conll', TCDATA_DIR + 'dev.conll',TCDATA_DIR + 'extra_train.conll'])
    extra_files = TCDATA_DIR + 'extra_train.conll'
    new_data = _fix_data(ent_tp_cnt,extra_files,iob=True)
    with open(TCDATA_DIR + 'extra_train_v2.conll','w') as f:
        for s in new_data:
            for x in s:
                f.write(x[0] + ' ' + x[1] + '\n')
            f.write('\n')   
    
    new_data = _fix_data(ent_tp_cnt,TCDATA_DIR + 'train.conll')
    with open(TCDATA_DIR + 'train_v2.conll','w') as f:
        for s in new_data:
            for x in s:
                f.write(x[0] + ' ' + x[1] + '\n')
            f.write('\n')  
    new_data = _fix_data(ent_tp_cnt,TCDATA_DIR + 'dev.conll')
    with open(TCDATA_DIR + 'dev_v2.conll','w') as f:
        for s in new_data:
            for x in s:
                f.write(x[0] + ' ' + x[1] + '\n')
            f.write('\n')

def create_extra_train_data():
    # 额外的训练数据
    # 数据来源：https://github.com/leodotnet/neural-chinese-address-parsing
    data = read_data([USERDATA_DIR + 'extra_data/train.txt', USERDATA_DIR +
                      'extra_data/dev.txt', USERDATA_DIR + 'extra_data/test.txt'])
    pattern = get_intersection_pattern()

    new_data = []
    for sentence in data:
        item = convert_data_format_v2(sentence)
        tags = convert_back_to_bio(item['label'],item['text'])

        # 对数据标签进行映射    
        new_tags = []
        for i,t in enumerate(tags):
            tt = t[2:]
            if tt in ['country','roomno','otherinfo','redundant']:
                new_tags.append('O')
            elif tt == 'person':
                new_tags.append(t[:2] + 'subpoi')
            elif tt == 'devZone':
                new_tags.append(t[:2] + 'devzone')
            elif tt in ['subRoad','subroad']:
                new_tags.append(t[:2] + 'road')
            elif tt in ['subRoadno','subroadno']:
                new_tags.append(t[:2] + 'roadno')
            else:
                new_tags.append(t) 

        # 处理distance
        new_tags,_ = convert_distance(item,new_tags)
        # 处理village_group
        new_tags,_ = convert_village(item, new_tags)
        # 处理intersection
        new_tags,_ = convert_intersection(item,new_tags,pattern)

        # 两个路之间的和字改成O
        spans = re.finditer('与|和',item['text'])
        for sp in spans:
            start,end = sp.span()
            if new_tags[start][2:]=='assist' and start > 0 and new_tags[start-1][2:] == 'road' and start < len(new_tags) and new_tags[start+1][2:] == 'road':
                new_tags[start] = 'O'
        
        # 去除噪声开头
        valid_start = ['B-prov','B-city','B-district','B-town','B-road','B-poi','B-devzone','B-community']
        for i,t in enumerate(new_tags):
            if t not in valid_start:
                continue
            break     
        new_tags = new_tags[i:]
        text = item['text'][i:]

        # 去除过短文本
        if len(text) <= 2:
            continue
        
        text = normalize(text)
        assert len(new_tags) == len(text),(text,new_tags,item,sentence)
        s = [(a,b) for a,b in zip(text,new_tags)]
        new_data.append(s)
    
    with open(TCDATA_DIR + 'extra_train.conll','w') as f:
        for s in new_data:
            for x in s:
                f.write(x[0] + ' ' + x[1] + '\n')
            f.write('\n')

if __name__ == '__main__':
    print('# create pretrain data')
    create_preatrain_data()
    print('# create extra data')
    create_extra_train_data()
    print('# fix wrong data')
    fix_data()
