import numpy as np

def F1(golden, preds):
    
    golden_dict = {}
    for gold in golden:
        sentid = gold['sentid']
        golden_dict[sentid] = {}
    for gold in golden:
        sentid = gold['sentid']
        start = gold['start_token']
        golden_dict[sentid][start] = {}
    for gold in golden:
        sentid = gold['sentid']
        start = gold['start_token']
        end = gold['end_token']
        golden_dict[sentid][start][end] = 1
    
    num_of_answers = len(golden) #np.mean([len(gold) for gold in golden])
    num_of_preds = len(preds)#np.mean([len(pred) for pred in preds])
    
    true_positive = 0
    for pred in preds:
        sentid = pred['sentid']
        start = pred['start_token']
        end = pred['end_token']
        if sentid in golden_dict:
            if start in golden_dict[sentid]:
                if end in golden_dict[sentid][start]:
                    true_positive += 1
    
    if true_positive == 0:
        precision = 0
        recall = 0
    else:
        precision = true_positive / num_of_preds
        recall = true_positive / num_of_answers
    
    
    if (precision + recall) == 0:
        if len(golden_dict) == 0:
            return (100,100,100)
        else:    
            return (0,0,0);
    f1 = 2 * precision * recall / (precision + recall)
    
    
    return precision, recall, f1

def accuracy(golden, preds):
    pass

    
