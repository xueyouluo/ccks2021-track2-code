import tensorflow as tf
import numpy as np
import re
import random
import json
import glob
import codecs
import os
import shutil
from conlleval import return_report
from tqdm import tqdm

np.random.seed(20190525)
random.seed(20190525)

def normalize(text):
    text = re.sub('[0-9]','0',text)
    text = re.sub('[a-zA-Z]','A',text)
    return text

def convert_data_format(sentence):
    word = ''
    tag = ''
    text = ''
    tag_words = []
    for i,(c,t) in enumerate(sentence):
        text += c
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
    
    return {"text":text,"label":entities}

def convert_back_to_bio(entities,text):
    tags = ['O'] * len(text)
    for t,words in entities.items():
        for w,spans in words.items():
            for s,e in spans:
                tags[s:e+1] = ['I-' + t] * (e-s+1)
                tags[s] = 'B-' + t
    return tags
    
def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def read_data(fnames, zeros=False, lower=False):
    '''
    Read all data into memory and convert to iobes tags.
    A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    Args:
        - fnames: a list of filenames contain the data
        - zeros: if we need to replace digits to 0s
    Return:
        - sentences: a list of sentnences, each sentence contains a list of (word,tag) pairs
    '''
    sentences = []
    sentence = []
    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
        sentence_num = 0
        num = 0
        print("read data from file {0}".format(fname))
        for line in codecs.open(fname, 'r', 'utf8'):
            num+=1
            line = line.rstrip()
            line = re.sub("\d+",'0',line) if zeros else line
            if not line:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence_num += 1
                    sentence = []
            else:
                # in case space is a word
                if line[0] == " ":
                    line = "$" + line[1:]
                    word = line.split()
                else:
                    word= line.split(' ')
                assert len(word) >= 2, print(fname,num,[word[0]],line)
                word[0] = word[0].lower() if lower else word[0]
                sentence.append(word)
        if len(sentence) > 0:
            sentence_num += 1
            sentences.append(sentence)
        print("Got {0} sentences from file {1}".format(sentence_num,fname))
    print("Read all the sentences from training files: {0} sentences".format(len(sentences)))
    return sentences

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def update_tag_scheme(sentences, tag_scheme='iobes', convert_to_iob=False):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if convert_to_iob:
            tags = iobes_iob(tags)
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            # we already did that in iob2 method
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def eval_ner(results, path, name):
      """
      Run perl script to evaluate model
      """
      if not os.path.exists(path):
          os.mkdir(path)
      output_file = os.path.join(path, name + "_ner_predict.utf8")
      with open(output_file, "w") as f:
          to_write = []
          for block in results:
              for line in block:
                  to_write.append(line + "\n")
              to_write.append("\n")

          f.writelines(to_write)
      eval_lines = return_report(output_file)
      f1 = float(eval_lines[1].strip().split()[-1])
      return eval_lines, f1

def convert_to_bio(tags):
    for i in range(len(tags)):
        t = tags[i]
        if t[0]=='B':
            j = i+1
            while j < len(tags) and tags[j][0] not in ['E','S']:
                j += 1
            if j >= len(tags):
                tags[i] = 'O'
            elif tags[j][0] == 'S':
                # error
                tags[i:j] = ['O'] * (j-i)
            elif tags[j][0] == 'E':
                tags[i+1:j+1] = ['I-span']*(j-i)
    tags = ['B'+t[1:] if t[0]=='S' else t for t in tags]
    return tags

def get_biaffine_pred_prob(text, span_scores, label_list):
    candidates = []
    for s in range(len(text)):
        for e in range(s,len(text)):
            candidates.append((s,e))
    
    top_spans = []
    for i,tp in enumerate(np.argmax(span_scores,axis=1)):
        if tp > 0:
            s,e = candidates[i]
            top_spans.append((s,e,label_list[tp],float(span_scores[i][tp])))

    top_spans = sorted(top_spans, key=lambda x:x[3], reverse=True)
    sent_pred_mentions = []
    for ns,ne,t,score in top_spans:
        for ts,te,_,_ in sent_pred_mentions:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                #for both nested and flat ner no clash is allowed
                break
            if (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                #for flat ner nested mentions are not allowed
                break
        else:
            sent_pred_mentions.append((ns,ne,t,score))
    return sent_pred_mentions

def get_biaffine_pred_ner(text, span_scores, is_flat_ner=True):
    candidates = []
    for s in range(len(text)):
        for e in range(s,len(text)):
            candidates.append((s,e))
    
    top_spans = []
    for i,tp in enumerate(np.argmax(span_scores,axis=1)):
        if tp > 0:
            s,e = candidates[i]
            top_spans.append((s,e,tp,span_scores[i]))

    top_spans = sorted(top_spans, key=lambda x:x[3][x[2]], reverse=True)
    
    # if not top_spans:
    #     # 无论如何找一个span
    #     # 这里是因为cluener里面基本上每句话都有实体，因此这样使用
    #     # 如果是真实的场景，可以去掉这部分
    #     tmp_span_scores = span_scores[:,1:]
    #     for i,tp in enumerate(np.argmax(tmp_span_scores,axis=1)):
    #         s,e = candidates[i]
    #         top_spans.append((s,e,tp+1,span_scores[i]))
    #     top_spans = sorted(top_spans, key=lambda x:x[3][x[2]], reverse=True)[:1]

    sent_pred_mentions = []
    for ns,ne,t,score in top_spans:
        for ts,te,_,_ in sent_pred_mentions:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                #for both nested and flat ner no clash is allowed
                break
            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                #for flat ner nested mentions are not allowed
                break
        else:
            sent_pred_mentions.append((ns,ne,t,[float(x) for x in score.flat]))
    return sent_pred_mentions

def get_biaffine_pred_ner_with_dp(text, span_scores, with_logits=True, threshold=0.1):
    candidates = []
    for s in range(len(text)):
        for e in range(s,len(text)):
            candidates.append((s,e))
            
    top_spans = {}
    for i,tp in enumerate(np.argmax(span_scores,axis=1)):
        if tp > 0:
            if not with_logits and span_scores[i][tp] < threshold:
                continue
            s,e = candidates[i]
            # if check_special_token(text[s:e+1]):
            #     continue
            top_spans[(s,e)] = (tp,span_scores[i][tp])


    if not top_spans:
        return []

    DAG = {}
    for k,v in top_spans:
        if k not in DAG:
            DAG[k] = []
        DAG[k].append(v)
     
    route = {}
    N = len(text)
    route[N] = (0,0)
    for idx in range(N-1,-1,-1):
        if with_logits:
            route[idx] = max(
                (top_spans.get((idx,x),[0,0])[1] + route[x+1][0],x)
                for x in DAG.get(idx,[idx])
            )
        else:
            route[idx] = max(
                ( np.log(max(top_spans.get((idx,x),[0,0])[1],1e-5)) + route[x+1][0],x)
                for x in DAG.get(idx,[idx])
            )
    
    start = 0
    spans = []
    while start < N:
        end = route[start][1]
        if (start,end) in top_spans:
            tp,score = top_spans[(start,end)]
            spans.append((start,end,tp,score))
        start = end + 1
    return spans

class SWAHook(tf.train.SessionRunHook):
    def __init__(self, swa_steps, start_swa_step, checkpoint_path):
        self.swa_steps = swa_steps
        self.start_swa_step = start_swa_step
        self.checkpoint_path = checkpoint_path
        self.pre_save_step = 0
        
    def begin(self):
        global_step = tf.train.get_global_step()
        self._global_step_tensor = tf.identity(global_step,"global_step_read")
        tvars = tf.trainable_variables()
        self.save_num = tf.Variable(initial_value=0, name="save_num",dtype=tf.float32,trainable=False)
        
        self.swa_vars = [
            tf.get_variable(
            name=tvar.name.split(":")[0] + "/swa",
            shape=tvar.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer()) for tvar in tvars]

        self.first_assign = tf.group([y.assign(x) for x,y in zip(tvars,self.swa_vars)] + [self.save_num.assign_add(1)])
        self.update = tf.group([y.assign((y*self.save_num + x)/(self.save_num+1)) for x,y in zip(tvars,self.swa_vars)]+ [self.save_num.assign_add(1)])
        to_save = {x.op.name:y for x,y in zip(tvars,self.swa_vars)}
        to_save[global_step.op.name] = global_step
        self.saver = tf.train.Saver(to_save,max_to_keep=1)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._global_step_tensor)
        if global_step >= self.start_swa_step:
            if self.pre_save_step == 0:
                run_context.session.run(self.first_assign)
                self.pre_save_step = global_step
            elif (global_step-self.pre_save_step) % self.swa_steps == 0:
                tf.logging.info('update swa')
                run_context.session.run(self.update)
                self.pre_save_step = global_step
              
    def end(self, session):
        global_step = session.run(self._global_step_tensor)
        self.saver.save(session,os.path.join(self.checkpoint_path,'model.ckpt'),global_step=global_step)

class BestF1Exporter(tf.estimator.Exporter):
    def __init__(self, input_fn, examples, label_list, max_seq_length, dp=False, name='f1_export'):
        self._name = name
        self.input_fn = input_fn
        self.predict_examples = examples
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self._best_eval_result = None
        self.dp = dp
    
    @property
    def name(self):
        return self._name

    def get_biaffine_result(self,estimator):
        final_results = []
        idx = 0
        for i,prediction in enumerate(tqdm(estimator.predict(input_fn=self.input_fn,yield_single_examples=True))):
            scores = prediction['score']
            offset = 0
            bz = prediction['batch_size']
            for j in range(bz):
                example = self.predict_examples[idx]
                text = example.text
                pred_text = example.text[:self.max_seq_length-2]
                size = len(pred_text) * (len(pred_text) + 1) // 2
                pred_score = scores[offset:offset+size]
                idx += 1
                offset += size
                if self.dp:
                    results = get_biaffine_pred_ner_with_dp(pred_text,pred_score)
                else:
                    results = get_biaffine_pred_ner(pred_text,pred_score)
                labels = {}
                for s,e,t,score in results:
                    span = text[s:e+1]
                    label = self.label_list[t]
                    item = [s,e]
                    if label not in labels:
                        labels[label] = {span:[item]}
                    else:
                        if span in labels[label]:
                            labels[label][span].append(item)
                        else:
                            labels[label][span] = [item]
                tags = convert_back_to_bio(labels,text)
                tags = [' '.join([c,t,p]) for c,t,p in zip(text,example.label,tags)]
                final_results.append(tags)
        return final_results

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        if not os.path.exists(export_path):
            tf.io.gfile.makedirs(export_path)
       
        final_results = self.get_biaffine_result(estimator)
        eval_lines, f1  = eval_ner(final_results,export_path,'eval')
        for line in eval_lines:
            tf.logging.info(line.rstrip())
        if self._best_eval_result is None or f1 > self._best_eval_result:
            tf.logging.info('Exporting a better model ({} instead of {}), ckp-path: {}'.format(
                    f1, self._best_eval_result,checkpoint_path))
            basename = None
            for name in glob.glob(checkpoint_path + '.*'):
                parts = os.path.basename(name).split('.')
                if len(parts) == 3:
                    parts[1] = parts[1].split('-')[0]
                filename = '.'.join(parts)
                basename = '.'.join(parts[:2])
                shutil.copy(name, os.path.join(export_path, filename))
            with open(os.path.join(export_path, "checkpoint"), 'w') as f:
                f.write("model_checkpoint_path: \"{}\"".format(basename))
            with open(os.path.join(export_path, "best.txt"), 'w') as f:
                f.write('Best f1: {}, path: {}\n'.format(f1,checkpoint_path))
            self._best_eval_result = f1
        else:
            tf.logging.info(
                'Keeping the current best model ({} instead of {}).'.format(
                    self._best_eval_result, f1))