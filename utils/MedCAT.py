from medcat.cat import CAT

class MedCAT_wrapper(object):
    def __init__(self, PATH):
        self.cat = CAT.load_model_pack(PATH)
    def match(self, sent, best_match=False, ignore_syntax=False):
        entities = self.cat.get_entities(sent)['entities']
        
        concepts = []
        for key in entities.keys():
            entity = entities[key]
            #print(entity)
            concept = {'start': entity['start'], 'end': entity['end'],
                       'cui': entity['cui'],
                       'similarity': entity['context_similarity'],
                       'semtypes': entity['type_ids'],
                       'term': entity['detected_name'],
                       'ngram': entity['source_value']}
            
            concepts.append([concept])
        return concepts
