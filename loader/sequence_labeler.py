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

