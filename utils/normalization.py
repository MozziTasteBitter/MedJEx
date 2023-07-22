import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM  
import numpy as np

class min_max_normalization(object):
    def __init__(self, weights):
        if weights:
            self.max = max(weights)
            self.min = min(weights)
        else:
            self.max = 1.0
            self.min = 0.0
            
    def get_normalized_results(self, val):
        ep = 1e-9
        return (val - self.min) / (self.max - self.min + ep)

class z_normalization(object):
    def __init__(self, weights):
        self.mean = np.mean(weights)
        self.std = np.std(weights)
        
    def get_normalized_results(self, val):
        ep = 1e-9
        return (val - self.mean) / (self.std + ep )
        

    
class Normalizer(object):
    def __init__(self,train_split,normalization_type):
        self.weights = {'term_frequency': [], 
                        'MLM_weight': []}
        
        self.normalizer = {}
        for train_data in train_split:
            concepts = train_data['UMLS_concepts']
            
            for index, concept in enumerate(concepts):
                for key in self.weights.keys():
                    self.weights[key].append(concept[key])
        self.normalization_type = normalization_type
        
        self.set_TF_normalizer()
        self.set_MLM_normalizer()
    def set_TF_normalizer(self):
        TF_weights = self.weights['term_frequency']
        
        if self.normalization_type == 'min_max':
            self.normalizer['term_frequency'] = min_max_normalization(TF_weights)
        else:
            self.normalizer['term_frequency'] = z_normalization(TF_weights)
        
    def set_MLM_normalizer(self):
        MLM_weight = self.weights['MLM_weight']
        
        if self.normalization_type == 'min_max':
            self.normalizer['MLM_weight'] = min_max_normalization(MLM_weight)
        else:
            self.normalizer['MLM_weight'] = z_normalization(MLM_weight)