import random
import os
from pprint import pprint as pprint
from typing import List, Optional

import numpy as np

import tqdm
import pandas
import sklearn

import torch
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

try: import transformers; 
except: 
    import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

from quickumls import QuickUMLS

import os, sys
#sys.path.append("../")
#print(sys.path)
#from utils.sequence_labeler import SequenceLabeler
#from sequence_labeler import SequenceLabeler


class SequenceLabeler(object):
  def __init__(self,entity_types = [], labeling_scheme='BIOES',longest_labeling_flag=False):
    label2id = {}
    if labeling_scheme=='BIOES':
      if not entity_types: 
        label2id = {'B':0,'I':1,'E':2,'S':3}
        index = 4
      else:
        index = 0
        for entiy_type in entiy_type:
          label2id['B-%s'%(entity_type)] = index + 0
          label2id['I-%s'%(entity_type)] = index + 1
          label2id['E-%s'%(entity_type)] = index + 2
          label2id['S-%s'%(entity_type)] = index + 3
          index += 4
      labelyid = label2id['O'] = index
    elif labeling_scheme=='BIO':
      if not entity_types: 
        label2id = {'B':0,'I':1}
        index = 2
      else:
        index = 0
        for entiy_type in entiy_type:
          label2id['B-%s'%(entity_type)] = index + 0
          label2id['I-%s'%(entity_type)] = index + 1
          index += 2
      labelyid = label2id['O'] = index


    self.id2label = {label2id[label]:label for label in label2id.keys()}
    self.label2id = label2id
    self.labeling_scheme = labeling_scheme
    self.longest_labeling_flag = longest_labeling_flag
    self.num_of_label = len(label2id)

  def labeling_validation_checker(self, tokens, labelings):
    labeling_checking_list = [None] * len(tokens)
    for start, end, uid in labelings:
      for i in range(start, end):
        if labeling_checking_list[i]:
          return False
        else:
          labeling_checking_list[i] = 1
    return True 

  def get_longest_for_nested_entities(self, tokens, labelings):
    token_dict = {i: [] for i in range(len(tokens)) }
    label_lens = {uid: end-start for start, end, uid in labelings}

    for start, end, uid in labelings:
      for i in range(start, end):
        token_dict[i].append(uid)

    unavailable_uid_list = []
    for i in range(len(tokens)):
      if len(token_dict[i]):
        if len(token_dict[i]) > 1:
          max_index_len = (-1, -1)
          for j, uid in enumerate(token_dict[i]):
            if label_lens[uid] > max_index_len[1]:
              max_index_len = (j, label_lens[uid])
          for j, uid in enumerate(token_dict[i]):
            if j != max_index_len[0]:
              unavailable_uid_list.append(uid)
    unavailable_uid_set = set(unavailable_uid_list)

    return [(start, end, uid) for start, end, uid in labelings if uid not in unavailable_uid_set]

  def get_labels(self,tokens, labelings, sentid=None):
    label2id = self.label2id
    labeling_scheme = self.labeling_scheme

    # Check if there are nested labels
    if not self.labeling_validation_checker(tokens, labelings):
      if self.longest_labeling_flag:
        labelings = self.get_longest_for_nested_entities(tokens, labelings)
      else:
        return None, None
    #if sentid==2332325: print(labelings);
    # Label with each scheme
    if labeling_scheme == 'BIOES':
      labels = self.BIOES_label(tokens, labelings)
    elif labeling_scheme =='BIO':
      labels = self.BIO_label(tokens, labelings)
    ids = [label2id[label] for label in labels]
    return labels, ids 

  def BIOES_label(self, tokens, labelings):
    ylabels = ['O']*len(tokens)
    for start, end, uid in labelings:
      if end - start <= 1:
        # assert ylabels[start] == 'O', "UID %s: Indexing error: duplicate labeling "%str(uid) 
        ylabels[start] = 'S'
      else:
        for i in range(start, end):
          # assert ylabels[i] == 'O', "UID %s: Indexing error: duplicate labeling "%str(uid)
          if i == start:
            ylabels[i] = 'B'
          else:
            ylabels[i] = 'I'
        ylabels[end-1] = 'E'
    return ylabels
  def BIO_label(self, tokens, labelings):
    ylabels = ['O']*len(tokens)
    for start, end, uid in labelings:
      if end - start <= 1:
        # assert ylabels[start] == 'O', "UID %s: Indexing error: duplicate labeling "%str(uid) 
        ylabels[start] = 'B'
      else:
        for i in range(start, end):
          # assert ylabels[i] == 'O', "UID %s: Indexing error: duplicate labeling "%str(uid)
          if i == start:
            ylabels[i] = 'B'
          else:
            ylabels[i] = 'I'
        ylabels[end-1] = 'I'
    return ylabels

  def BIOES_decode(self, tokens, labels, tokenizer = None):
    # Output [{entity_type="", text = "", entity_token_span=(start,end)}]
    assert len(tokens) != len(labels), "the length of tokens and labels should be same."
    entity_list = []

    inner = False
    for i, (token, label) in enumerate(zip(tokens, labels)):
      label_type = label[0]
      if len(label) > 0: 
        entity_type = label[2:]
      else:
        entity_type = ""
      
      # Type 1, not inner 
      if not inner:
        if label_type == 'B':
          inner = (i, entity_type)
        elif label_type == 'S':
          entity_list.append((i, i+1, entity_type))
        else: continue;
      else:
        if label_type == 'B' or label_type == 'S':
          inner = False; continue;
        elif inner[1] != entity_type:
          inner = False; continue;
        elif label_type == 'E':
          start = inner[0]; end = i+1
          inner = False; entity_list.append((start, end, entity_type))
        else:
          continue
    
    entities = []
    for entity in entity_list:
      start = entity[0]; end = entity[1]; entity_type = entity[2]
      if not tokenizer:
        text = " ".join(tokens[start: end])
      else:
        text = tokenizer.convert_tokens_to_string(tokens[start: end])
      entities.append({ 'entity_type': entity_type,
                        'entity_token_span': (start, end),
                        'start_token': start,
                        'end_token': end,
                        'text:': text})

    return entities

  def BIO_decode(self, tokens, labels, tokenizer = None):
    # Output [{entity_type="", text = "", entity_token_span=(start,end)}]
    assert len(tokens) == len(labels), "the length of tokens and labels should be same."
    entity_list = []

    inner = False
    for i, (token, label) in enumerate(zip(tokens, labels)):
      label_type = label[0]
      if len(label) > 0: 
        entity_type = label[2:]
      else:
        entity_type = ""
      
      # Type 1, not inner 
      if not inner:
        if label_type == 'B':
          inner = (i, entity_type)
        # elif label_type == 'S':
        #   entity_list.append((i, i+1, entity_type))
        else: continue;
      else:
        if label_type == 'B':
          start = inner[0]; end = i
          entity_list.append((start, end, entity_type))
          inner = (i, entity_type)
        elif inner[1] != entity_type:
          inner = False; continue;
        elif label_type == 'O':
          start = inner[0]; end = i
          inner = False; entity_list.append((start, end, entity_type))
        else:
          continue
    if inner:
      start = inner[0]; end = i+1
      inner = False; entity_list.append((start, end, entity_type))
    entities = []
    for entity in entity_list:
      start = entity[0]; end = entity[1]; entity_type = entity[2]
      if not tokenizer:
        text = " ".join(tokens[start: end])
      else:
        text = tokenizer.convert_tokens_to_string(tokens[start: end])
      entities.append({ 'entity_type': entity_type,
                        'entity_token_span': (start, end),
                        'start_token': start,
                        'end_token': end,
                        'text:': text})

    return entities



  def BIOES_ids2entities(self, ids, tokenizer = None):
    # Output [{entity_type="", text = "", entity_token_span=(start,end)}]
    # assert len(tokens) == len(labels), "the length of tokens and labels should be same."
    entity_list = []

    inner = False
    for i, label_id in enumerate(ids):
      label = self.id2label[label_id]
      label_type = label[0]
      if len(ids) > 0: 
        entity_type = label[2:]
      else:
        entity_type = ""
      
      # Type 1, not inner 
      if not inner:
        if label_type == 'B':
          inner = (i, entity_type)
        elif label_type == 'S':
          entity_list.append((i, i+1, entity_type))
        else: continue;
      else:
        if label_type == 'B' or label_type == 'S':
          inner = False; continue;
        elif inner[1] != entity_type:
          inner = False; continue;
        elif label_type == 'E':
          start = inner[0]; end = i+1
          inner = False; entity_list.append((start, end, entity_type))
        else:
          continue
    entities = []
    for entity in entity_list:
      start = entity[0]; end = entity[1]; entity_type = entity[2]
      entities.append({ 'entity_type': entity_type,
                        'entity_token_span': (start, end),
                        'start_token': start,
                        'end_token': end})

    return entities

  def BIO_ids2entities(self, ids, tokenizer = None):
    # Output [{entity_type="", text = "", entity_token_span=(start,end)}]
    # assert len(tokens) == len(labels), "the length of tokens and labels should be same."
    entity_list = []

    inner = False
    for i, label_id in enumerate(ids):
      label = self.id2label[label_id]
      label_type = label[0]
      if len(ids) > 0: 
        entity_type = label[2:]
      else:
        entity_type = ""
      
      # Type 1, not inner 
      if not inner:
        if label_type == 'B':
          inner = (i, entity_type)
        # elif label_type == 'S':
        #   entity_list.append((i, i+1, entity_type))
        else: continue;
      else:
        if label_type == 'B':
          start = inner[0]; end = i
          entity_list.append((start, end, entity_type))
          inner = (i, entity_type)
        elif inner[1] != entity_type:
          inner = False; continue;
        elif label_type == 'O':
          start = inner[0]; end = i
          inner = False; entity_list.append((start, end, entity_type))
        else:
          continue
    if inner:
      start = inner[0]; end = i
      inner = False; entity_list.append((start, end, entity_type))
    entities = []
    for entity in entity_list:
      start = entity[0]; end = entity[1]; entity_type = entity[2]
      entities.append({ 'entity_type': entity_type,
                        'entity_token_span': (start, end),
                        'start_token': start,
                        'end_token': end})

    return entities






def load_file(note_aid_path):
  note_aid_df = pandas.read_csv(note_aid_path)
  #sentence_df = pandas.read_csv(sentence_path)
  
  note_aid_data_dict = {'uid2sentid_dict':{},
                        'sentid2uids_dict':{}, 
                        'data_dict':{},
                        'sent_dict':{},}
  #for index in tqdm.tqdm(range(len(note_aid_df))):
  for index in range(len(note_aid_df)):
    # Get labeling data
    row = note_aid_df.iloc[index]
    uid = str(index); sentence_id = str(row['sentenceid'])
    start = row['start']; end = row['end']
    term = row['term']; definition = row['def']
    #print(definition)
    # Get sentence dictionary
    sent = row['sentence']
    text_term = " ".join(sent.split()[start:end])
    
    # Set sentence dictionarty
    note_aid_data_dict['sent_dict'][sentence_id] = sent
    # Data checking & Ignore data code
    # If a term is not defined then igore it.
    # print(definition)
    if row['do_not_define']==1:
      continue
    elif str(definition) == 'nan':
      #print('definition is nan')
      continue
    # If start index is small or same with end index ignore
    try: assert start < end
    except: 
      print("UID %s: start index (%d) must smaller than end (%d)"%(str(uid),start, end))
      continue;
    # If start index is wrong then ignore
    try: assert start >= 0
    except:
      print("UID %s: start index (%d) error"%(str(uid), start))
      continue;
    # If end index is wrond then ignore
    try: assert end <= len(sent.split())
    except: 
      print("UID %s: end index (%d) error it should be less than words length (%d)"%(str(uid), end, len(sent.split())))
      continue;
    try: assert start < len(sent.split())
    except: 
      print("UID %s: end index (%d) error it should be less than words length (%d)"%(str(uid), end, len(sent.split())))
      continue;
    
    

    

    

    # Set dictionaries for indexing uid and sentence id
    note_aid_data_dict['uid2sentid_dict'][uid] = sentence_id
    if sentence_id not in note_aid_data_dict['sentid2uids_dict']:
      note_aid_data_dict['sentid2uids_dict'][sentence_id] = []
    note_aid_data_dict['sentid2uids_dict'][sentence_id].append(uid)
    


    # Set data dictionary
    data = {'sent_id': sentence_id,
            'start': start, 
            'end': end,
            'definition': definition,
            'term': term,
            'text_term': text_term,
            'definition': definition}
    note_aid_data_dict['data_dict'][uid] = data

    

  return note_aid_data_dict

def load_ner_file(ner_data_file):
  fin = open(ner_data_file)
  BIO_labeler = SequenceLabeler(labeling_scheme='BIO',longest_labeling_flag=True)  
    
    #note_aid_df = pandas.read_csv(note_aid_path)
  #sentence_df = pandas.read_csv(sentence_path)
  
  note_aid_data_dict = {'uid2sentid_dict':{},
                        'sentid2uids_dict':{}, 
                        'data_dict':{},
                        'sent_dict':{},}
  #for index in tqdm.tqdm(range(len(note_aid_df))):
    
  sentences = []
  sentences_labels = []
    
  sflag = False
  words = []; labels = []
  for line in fin:
    # Get labeling data
    line = line.strip();
    
    if not line:
        if not sflag:
            continue;
        else:
            sentences.append(words)
            sentences_labels.append(labels)
            words = []; labels = []
            sflag = False
    else:
        sflag = True
        word, label = line.split('\t')
        words.append(word); labels.append(label)
  if sflag:
    sentences.append(words)
    sentences_labels.append(labels)
    words = []; labels = []
    sflag = False
    
  sent_id = 0
  for words, labels in zip(sentences, sentences_labels):
    # Set dictionaries for indexing uid and sentence id
    sentence_id = str(sent_id)
    note_aid_data_dict['sent_dict'][sentence_id] = " ".join(words)
    
    entities = BIO_labeler.BIO_decode(words, labels)
    # print(labels)
    # print(entities)
    for i, entity in enumerate(entities):
        uid = "%s_%d"%(sentence_id, i)
        note_aid_data_dict['uid2sentid_dict'][uid] = sentence_id
        if sentence_id not in note_aid_data_dict['sentid2uids_dict']:
          note_aid_data_dict['sentid2uids_dict'][sentence_id] = []
        note_aid_data_dict['sentid2uids_dict'][sentence_id].append(uid)


        start = entity['start_token']; end = entity['end_token']
        term = entity['text:']; text_term = entity['text:'];
        definition = "."




        # Set data dictionary
        data = {'sent_id': sentence_id,
                'start': start, 
                'end': end,
                'definition': definition,
                'term': term,
                'text_term': text_term,
                'definition': definition}
        note_aid_data_dict['data_dict'][uid] = data
    sent_id += 1
    
  fin.close()
  return note_aid_data_dict





def load_data(note_aid_data_dict, tokenizer, labeler, UMLS_matcher = None, UMLS_labeler = None):
  # assert not QuickUMLS_PATH and UMLS_labeler

  sent_dict = note_aid_data_dict['sent_dict']
  data_dict = note_aid_data_dict['data_dict']
  uid2sentid_dict = note_aid_data_dict['uid2sentid_dict']
  sentid2uids_dict = note_aid_data_dict['sentid2uids_dict']
  
  # tokenize and label answers for each sentence
  nested = 0

  processed_data_dict = {}
  token_len_list = []
  
  matcher = UMLS_matcher
  # if QuickUMLS_PATH:
  #   matcher = QuickUMLS(QuickUMLS_PATH, overlapping_criteria="score")

  for sentid in tqdm.tqdm(list(sent_dict.keys())):
    
    sent = sent_dict[sentid]
    if sentid in sentid2uids_dict:
        uids = sentid2uids_dict[sentid]
    else:
        uids = []
    
            
    
    word_index2token_indices = {}
    words = sent.split()
    
    
    char_index2word_index = {}
    char_index = 0
    for word_index, word in enumerate(words):
      for char in word:
        char_index2word_index[char_index] = word_index
        char_index += 1
      char_index += 1
    
    token_ids = []
    token_index = 1
    for word_index, word in enumerate(words):
      word_tokens = tokenizer.encode(word ,add_special_tokens=False)
      word_index2token_indices[word_index] = []

      for word_token in word_tokens:
        token_ids.append(word_token)
        word_index2token_indices[word_index].append(token_index)
        token_index += 1
    cls_token_ids = tokenizer.cls_token_id
    sep_token_ids = tokenizer.sep_token_id

    token_ids = [cls_token_ids] + token_ids + [sep_token_ids]
    tokens = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in token_ids]

    labelings = []
    entities = []
    for uid in uids:
      data = data_dict[uid]
      start = data['start']; end = data['end']
      #print(word_index2token_indices)
      try:
        start_token_index = min(word_index2token_indices[start])
      except:
        print(sentid)
        print(sent)
        print(data) 
        print(word_index2token_indices)
      end_token_index = max(word_index2token_indices[end-1]) + 1
      labelings.append((start_token_index, end_token_index, uid))


      entity = {'uid':uid,
                'start_token': start_token_index, 
                'end_token': end_token_index,
                'term': data['term']}
      entities.append(entity)
      token_len_list.append((end_token_index - start_token_index + 1,uid))

    ylabels, yids = labeler.get_labels(tokens, labelings)

    if not ylabels:
      print("Sentence Id %s has more than one invalid nested entities."%sentid)
      print(word_index2token_indices)
      print(labelings)
      nested += 1
      continue
    
    
    
    
 
    input = {'sentid': sentid,
             'ylabels': ylabels,
             'yids': yids,
             'text': sent,
             'words': words,
             'tokens': tokens,
             'token_ids': token_ids,
             'entities':entities,
             'word_index2token_indices':word_index2token_indices,
             'char_index2word_index':char_index2word_index,}
    
    
    
    #   {'start': 4,
    # 'end': 8,
    # 'ngram': 'ulna',
    # 'term': 'ulna',
    # 'cui': 'C0041600',
    # 'similarity': 1.0,
    # 'semtypes': {'T023'},
    # 'preferred': 0}
    if UMLS_matcher:
      concepts = matcher.match(sent, best_match=False, ignore_syntax=True)
      UMLS_concepts = []
      labelings = []
      for index, concept in enumerate(concepts):
        # set as a first concept among the candidates
        concept = concept[0]
        cui = concept['cui']; term = concept['term']; semtypes=concept['semtypes']
        start_char = concept['start']; end_char = concept['end'] -1 ;
        
        try:
            start_token_index = min(word_index2token_indices[char_index2word_index[start_char]])
        except:
            word_index2token_indices
            
        end_token_index = max(word_index2token_indices[char_index2word_index[end_char]]) + 1
        
        labelings.append((start_token_index, end_token_index, cui))
        
        UMLS_concept = {'uid':str(sentid)+'_'+str(index),
                        'cui': cui,
                        'term': term,
                        'semtypes': semtypes,
                        'start_token': start_token_index, 
                        'end_token': end_token_index}
        UMLS_concepts.append(UMLS_concept)
        
      input['UMLS_concepts'] = UMLS_concepts
    
      #UMLS_ylabels, UMLS_yids = UMLS_labeler.get_labels(tokens, labelings)
      #input['UMLS_yids'] = UMLS_yids
      #input['UMLS_ylabels'] = UMLS_ylabels
       
      UMLS_bin_dim = len(UMLS_labeler.id2label)
      token_len = len(tokens)
      UMLS_bin_representation = np.zeros((token_len, UMLS_bin_dim))
    
      for labeling in labelings:
        UMLS_ylabels, UMLS_yids = UMLS_labeler.get_labels(tokens, [labeling])
        for token_id, yid in enumerate(UMLS_yids):
            UMLS_bin_representation[token_id][UMLS_labeler.label2id['O']] = 1.0
            if yid != UMLS_labeler.label2id['O']:
                UMLS_bin_representation[token_id][yid] = 1.0
                UMLS_bin_representation[token_id][UMLS_labeler.label2id['O']] = 0.0
      
      
      input['UMLS_bin_representation'] = UMLS_bin_representation
    processed_data_dict[sentid] = input
    
  print("The num of nested sentences is %d"%nested)
  #print(token_len_list)
  length_list = [length for length, uid in token_len_list]
  token_sum = sum(length_list); entity_num = len(length_list);
  print(token_sum)

  #print("Average number of token is: %.2f"%(float(token_sum)/entity_num))
  # max_index = np.argmax(length_list)
  # max_len = token_len_list[max_index][0]; max_id = token_len_list[max_index][1]
  # print(max_len, max_id)
  # print("The length of the tokens with the longest entity: %d which uid is %s"%(max_len, max_id))

  return processed_data_dict

