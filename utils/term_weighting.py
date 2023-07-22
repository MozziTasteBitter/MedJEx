import copy
from tqdm import tqdm

import wordfreq
from wordfreq import word_frequency, zipf_frequency

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM  

class TermFrequency(object):
    def __init__(self):
        pass
    
    def get_weight(self, term, mode='zipf', lang = 'en'):
        # word freq = word probability
        if mode == 'word':
            
            term_weight = word_frequency(term, lang, wordlist='best', minimum=0.0)
        
        # Zipf freq = log10(bilion * word_frequency)
        elif mode == 'zipf':
            term_weight = zipf_frequency(term, lang, wordlist='best', minimum=0.0)
        
        else:
            term_weight = None
        
        return term_weight
        
    def get_weights(self, processed_data, target_entities = 'entities'):
        temp_processed_data = []
        for processed_sentence in tqdm(processed_data):
            token_ids = processed_sentence['token_ids']; 
            entities = processed_sentence[target_entities]
            
            temp_entities = []
            for entity in entities:
                term = entity['term']
                entity['term_frequency'] = self.get_weight(term, mode = 'zipf', lang = 'en')
                temp_entities.append(entity)
            processed_sentence[target_entities] = temp_entities
            temp_processed_data.append(processed_sentence)
        return temp_processed_data
    
    
"""
Class for get BERT-based term weighting
__init__(model_name = ""): 1) initialize tokenizer and model, 2) set [MASK]ing func
"""

class MLM_weight(object):
    def __init__(self, model_name, mode='mask'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        pretrained_model_config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_id = self.tokenizer.mask_token_id
        
        
        # self.preprocessor = self.set_preprocessor()
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # set mode
        # mode == (mask, flat)
        self.mode = mode
        
    def mask_sentence_generator(self,token_ids, entities):
        mask_id = self.mask_id
        
        masked_token_sentences = []
        for entity in entities:
            tokens = copy.copy(token_ids)
            start_token = entity['start_token']; end_token = entity['end_token']; #uid = entity['uid']
            if self.mode == 'mask':
                masked_sentences = tokens[:start_token]  + [mask_id] * (end_token - start_token) + tokens[end_token:]
            else:
                masked_sentences = tokens
            # entity['masked_sentences'] = masked_sentences
            masked_token_sentences.append(masked_sentences)
            
        
        return masked_token_sentences    
        
    def get_masked_weights(self, masked_token_sentences, entities, golden_tokens):
        def make_inputs(masked_token_sentences, golden_tokens):
            device = self.device
            # print(masked_token_sentences)
            token_len = len(masked_token_sentences[0])
            
            input_ids = [masked_token_sentence for masked_token_sentence in masked_token_sentences]
            attention_mask = [[1]*token_len for masked_token_sentence in masked_token_sentences] 
            token_type_ids = [[0]*token_len for masked_token_sentence in masked_token_sentences] 
            labels = [golden_tokens for masked_token_sentence in masked_token_sentences]
            
            
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            token_type_ids = torch.tensor(token_type_ids).to(device)
            labels = torch.tensor(labels).to(device)
            
            return input_ids, attention_mask, token_type_ids, labels
        
        def nll(masked_logits, golden, spans):
            start_token, end_token = spans
            # masked_logits.cpu().data; golden.cpu().data; 
            # masked_logits = masked_logits[start_token:end_token]
            # golden = golden[start_token:end_token]
            softmax = torch.nn.Softmax(dim=-1)

            
            masked_probs = softmax(masked_logits)
            
           
            golden = [[gold] for gold in golden]
            golden = torch.tensor(golden)
            # print(golden)
            masked_answer_probs = torch.gather(masked_probs, 1, golden)[start_token: end_token]
            
            
            # print(self.tokenizer.decode(torch.squeeze(golden[start_token: end_token], -1)))
            # print(torch.log(masked_answer_probs))
            # print()
            
            masked_negative_log_likelihood = -1 * torch.mean(torch.log(masked_answer_probs))
            # masked_negative_log_likelihood.item() 
            
            return masked_negative_log_likelihood.item()
        with torch.no_grad():    
            model = self.model

            input_ids, attention_mask, token_type_ids, labels = make_inputs(masked_token_sentences, golden_tokens)

            outputs = model(input_ids, attention_mask=attention_mask,labels=labels, token_type_ids = token_type_ids)
            
            try:
                loss = outputs[0].item(); 
            except:
                print(input_ids.shape)
                #print(outputs)
            logits = outputs[1].cpu().data;

            weights = []
            # print(logits)
            for masked_token_sentence, masked_logits in zip(entities,logits):
                start_token = masked_token_sentence['start_token']; end_token = masked_token_sentence['end_token']
                masked_nll = nll(masked_logits, golden_tokens, (start_token, end_token))
                weight = masked_nll
                weights.append(weight)
            # print(outputs)
            # print(outputs[1].shape)
            # print(self.tokenizer.decode(golden_tokens))
            # print(self.tokenizer.decode(input_ids[1]))
            del(input_ids); del(attention_mask); del(token_type_ids); del(labels); del(outputs)
            return weights
    
    def get_weights(self, processed_data, target_entities = 'UMLS_entities', ignore = False):
        temp_processed_data = []
        for processed_sentence in tqdm(processed_data):
            
            token_ids = processed_sentence['token_ids']; 
            entities = processed_sentence[target_entities];
            if entities:
                # temp_entities = [copy.copy(entity )for entity in entities]
                
                if not ignore:
                    masked_token_sentences = self.mask_sentence_generator(token_ids, entities)
                    weights = self.get_masked_weights(masked_token_sentences, entities, token_ids)
                else: 
                    weights = [0.0 for entity in entities]
                temp_entities = []
                for weight, entity in zip(weights, entities):
                    entity['MLM_weight'] = weight
                    # entity['MLM_weight'] = 0.0
                    temp_entities.append(entity)
                processed_sentence[target_entities] = temp_entities
                temp_processed_data.append(processed_sentence)
            else: 
                temp_processed_data.append(processed_sentence)
        torch.cuda.empty_cache()
        return temp_processed_data