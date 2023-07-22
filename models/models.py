import torch
import torch.nn as nn
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import transformers
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaTokenizer, RobertaPreTrainedModel,RobertaForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig



from torchcrf import CRF

import random
manual_seed = 0
random.seed(manual_seed)
# torch.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

BertLayerNorm = torch.nn.LayerNorm

class BERT_MLP(BertPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        
        try:
            self.num_of_additional_features = num_of_additional_features
        except:
            self.num_of_additional_features = 0

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        
        self.num_of_additional_features = num_of_additional_features
        
        self.feedforward_layer = nn.Linear(config.hidden_size + self.num_of_additional_features, 
                                           config.hidden_size + self.num_of_additional_features)
        self.classifier = nn.Linear(config.hidden_size + self.num_of_additional_features, 
                                    config.num_labels)

        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tokenizer=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        if self.num_of_additional_features:
            sequence_output = torch.cat((sequence_output, additional_features), -1)
        
        # print(sequence_output.shape)
        # output_hidden = sequence_output
        
        output_hidden = self.feedforward_layer(sequence_output)
        output_hidden = torch.tanh(output_hidden)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


    
    
class BertForTokenClassification(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = softmax(logits)
        preds = torch.argmax(probs, dim=-1).cpu().detach().numpy()
        output = (preds,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class BertCRFs(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(logits, labels), self.crf.decode(logits)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return (None, sequence_of_tags)
        
        
class BertMLPCRFs(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=0):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        try:
            self.num_of_additional_features = num_of_additional_features
        except:
            self.num_of_additional_features = 0
        
        self.num_of_additional_features = num_of_additional_features
        
        self.feedforward_layer = nn.Linear(config.hidden_size + self.num_of_additional_features, 
                                           config.hidden_size + self.num_of_additional_features)
        self.classifier = nn.Linear(config.hidden_size + self.num_of_additional_features, 
                                    config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        if self.num_of_additional_features:
            sequence_output = torch.cat((sequence_output, additional_features), -1)
        
        # print(sequence_output.shape)
        # output_hidden = sequence_output
        
        output_hidden = self.feedforward_layer(sequence_output)
        output_hidden = torch.tanh(output_hidden)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden + sequence_output)
        #logits = self.classifier(output_hidden)
        
        boolean_attention_mask = attention_mask.type(torch.bool)
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(logits, labels, boolean_attention_mask), self.crf.decode(logits, boolean_attention_mask)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, boolean_attention_mask)
            return (None, sequence_of_tags)
        
        
        
class EarlyFusionBertMLPCRF(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=0):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        try:
            self.num_of_additional_features = num_of_additional_features
        except:
            self.num_of_additional_features = 0
        
        self.num_of_additional_features = num_of_additional_features
        if self.num_of_additional_features:
            self.embedding_layer = nn.Linear(self.num_of_additional_features, 
                                               config.hidden_size)
            self.embedding_layer_1 = nn.Linear(config.hidden_size, 
                                               config.hidden_size)
            # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
            #                                    config.hidden_size)
        
        self.feedforward_layer = nn.Linear(config.hidden_size, 
                                           config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 
                                    config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.embeddings = self.bert.embeddings
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        inputs_embeds += self.embeddings.token_type_embeddings(token_type_ids)
        
        #inputs_embeds += self.embeddings.position_embeddings(position_ids)
        if self.num_of_additional_features:
            #inputs_embeds = torch.cat((inputs_embeds, additional_features), -1)
            #inputs_embeds += self.embedding_layer(additional_features)
            embedded_additional_feature = torch.tanh(self.embedding_layer(additional_features))
            # embedded_additional_feature = self.LayerNorm(embedded_additional_feature)
            embedded_additional_feature = self.dropout(embedded_additional_feature)
            inputs_embeds += torch.tanh(self.embedding_layer_1(embedded_additional_feature)) 
            
            
            
        outputs = self.bert(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        
        # print(sequence_output.shape)
        # output_hidden = sequence_output
        
        output_hidden = self.feedforward_layer(sequence_output)
        output_hidden = torch.tanh(output_hidden)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden + sequence_output)
        #logits = self.classifier(output_hidden)
        
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(logits, labels), self.crf.decode(logits)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return (None, sequence_of_tags) 
        
        
        
class EarlyAndHiddenConcatBertMLPCRF(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # try:
        #     self.num_of_additional_features = num_of_additional_features
        # except:
        #     self.num_of_additional_features = 0
        
        # self.num_of_additional_features = num_of_additional_features
        # if self.num_of_additional_features:
        #     self.embedding_layer = nn.Linear(self.num_of_additional_features, 
        #                                        config.hidden_size)
        #     self.embedding_layer_1 = nn.Linear(config.hidden_size, 
        #                                        config.hidden_size)
        #     # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
        #     #                                    config.hidden_size)
        self.feedforward_layer = nn.Linear(config.hidden_size, 
                                           config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 
                                    config.num_labels)
        self.softmax = nn.Softmax(dim = -1)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        

        if num_of_additional_features:
            self.num_of_binary_features = num_of_additional_features['num_of_binary_features']
            self.num_of_weighted_features = num_of_additional_features['num_of_weighted_features']
            
            if self.num_of_binary_features:
                self.embedding_layer = nn.Linear(self.num_of_binary_features, 
                                                   config.hidden_size)
                self.embedding_layer_1 = nn.Linear(config.hidden_size, 
                                                   config.hidden_size)
                # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
                #                                    config.hidden_size)
            if self.num_of_weighted_features:
                self.feedforward_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                                                   config.hidden_size)
                # self.weighted_feature_hidden_layer = nn.Linear(self.num_of_weighted_features, 
                #                                    config.hidden_size)
                # self.weighted_feature_classifier = nn.Linear(config.hidden_size, 
                #                                             config.num_labels)
            
        
        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.embeddings = self.bert.embeddings
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        inputs_embeds += self.embeddings.token_type_embeddings(token_type_ids)
        
        #inputs_embeds += self.embeddings.position_embeddings(position_ids)
        binary_features = additional_features['binary_features']
        
        if self.num_of_binary_features:
            #inputs_embeds = torch.cat((inputs_embeds, additional_features), -1)
            #inputs_embeds += self.embedding_layer(additional_features)
            embedded_additional_feature = torch.tanh(self.embedding_layer(binary_features))
            # embedded_additional_feature = self.LayerNorm(embedded_additional_feature)
            embedded_additional_feature = self.dropout(embedded_additional_feature)
            inputs_embeds += torch.tanh(self.embedding_layer_1(embedded_additional_feature)) 
        
            
            
        outputs = self.bert(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        #logits = self.classifier(output_hidden)
        if self.num_of_weighted_features:
            weighted_features = additional_features['weighted_features']
            sequence_output = torch.cat((sequence_output, weighted_features), -1)
            # weighted_feature_hidden = torch.tanh(self.weighted_feature_hidden_layer(weighted_features))
            # weighted_feature_logits = self.weighted_feature_classifier(weighted_feature_hidden)
         
            
        output_hidden = self.feedforward_layer(sequence_output)
        output_hidden = torch.tanh(output_hidden)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden)
        emissions = logits
        
   
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(emissions, labels), self.crf.decode(emissions, attention_mask)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, attention_mask)
            return (None, sequence_of_tags) 
        

class EarlyAndLateFusionBertMLPCRF(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=None, weighted_feature_gate = True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # try:
        #     self.num_of_additional_features = num_of_additional_features
        # except:
        #     self.num_of_additional_features = 0
        
        # self.num_of_additional_features = num_of_additional_features
        # if self.num_of_additional_features:
        #     self.embedding_layer = nn.Linear(self.num_of_additional_features, 
        #                                        config.hidden_size)
        #     self.embedding_layer_1 = nn.Linear(config.hidden_size, 
        #                                        config.hidden_size)
        #     # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
        #     #                                    config.hidden_size)
        self.feedforward_layer = nn.Linear(config.hidden_size, 
                                           config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 
                                    config.num_labels)
        self.softmax = nn.Softmax(dim = -1)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.weighted_feature_gate = weighted_feature_gate
        if num_of_additional_features:
            self.num_of_binary_features = num_of_additional_features['num_of_binary_features']
            self.num_of_weighted_features = num_of_additional_features['num_of_weighted_features']
            
            if self.num_of_binary_features:
                self.embedding_layer = nn.Linear(self.num_of_binary_features, 
                                                   config.hidden_size)
                self.embedding_layer_1 = nn.Linear(config.hidden_size, 
                                                   config.hidden_size)
                # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
                #                                    config.hidden_size)
            if self.num_of_weighted_features:
                self.WeightedFeatureLayerNorm = torch.nn.LayerNorm(config.hidden_size + self.num_of_weighted_features, eps=config.layer_norm_eps)
                # self.feedforward_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                #                                    config.hidden_size)
                self.weighted_feature_hidden_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                                                   config.hidden_size)
                self.weighted_feature_classifier = nn.Linear(config.hidden_size,
                                                            config.num_labels)
                if self.weighted_feature_gate:
                    self.gate_hidden_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                                                       config.hidden_size)
                    self.gate_layer = nn.Linear(config.hidden_size, 1)
        else: 
            self.num_of_binary_features = 0
            self.num_of_weighted_features = 0
        self.init_weights()
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.embeddings = self.bert.embeddings
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        inputs_embeds += self.embeddings.token_type_embeddings(token_type_ids)
        
        
        
        if self.num_of_binary_features:
            #inputs_embeds += self.embeddings.position_embeddings(position_ids)
            binary_features = additional_features['binary_features']
            #inputs_embeds = torch.cat((inputs_embeds, additional_features), -1)
            #inputs_embeds += self.embedding_layer(additional_features)
            embedded_additional_feature = torch.tanh(self.embedding_layer(binary_features))
            # embedded_additional_feature = self.LayerNorm(embedded_additional_feature)
            embedded_additional_feature = self.dropout(embedded_additional_feature)
            inputs_embeds += torch.tanh(self.embedding_layer_1(embedded_additional_feature)) 
            
            
            
        outputs = self.bert(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        #logits = self.classifier(output_hidden)
        
            
        output_hidden = self.feedforward_layer(sequence_output)
        
        
        
        output_hidden = torch.tanh(self.LayerNorm(output_hidden))
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden)
        emissions = logits
        boolean_attention_mask = attention_mask.type(torch.bool)
        if self.num_of_weighted_features:
            
            weighted_features = additional_features['weighted_features']
            weighted_features = self.WeightedFeatureLayerNorm(torch.cat((sequence_output, weighted_features), -1))
            weighted_feature_hidden = torch.tanh(self.LayerNorm(self.weighted_feature_hidden_layer(weighted_features)))
            weighted_feature_hidden = self.dropout(weighted_feature_hidden)
            weighted_feature_logits = self.weighted_feature_classifier(weighted_feature_hidden)
            #weighted_feature = torch.sigmoid(weighted_feature_logits)
            #weighted_feature = weighted_feature_logits

            if not self.weighted_feature_gate:
                emissions = emissions + weighted_feature_logits
            else:
                gate_input =  self.WeightedFeatureLayerNorm(weighted_features)
                gate_hidden = self.LayerNorm(self.gate_hidden_layer(gate_input))
                gate_hidden = self.dropout(torch.tanh(gate_hidden))
                weight_feature_gate = torch.sigmoid(self.gate_layer(gate_hidden))
                
                emissions = weight_feature_gate * emissions + (1- weight_feature_gate) * weighted_feature_logits
        
   
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(emissions, labels, boolean_attention_mask), self.crf.decode(emissions, boolean_attention_mask)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, boolean_attention_mask)
            return (None, sequence_of_tags)         

        
#from transformers import RobertaModel, RobertaTokenizer, RobertaPreTrainedModel, AdamW, get_linear_schedule_with_warmup
class EarlyAndLateFusionrobertaMLPCRF(RobertaPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=None, weighted_feature_gate = True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # try:
        #     self.num_of_additional_features = num_of_additional_features
        # except:
        #     self.num_of_additional_features = 0
        
        # self.num_of_additional_features = num_of_additional_features
        # if self.num_of_additional_features:
        #     self.embedding_layer = nn.Linear(self.num_of_additional_features, 
        #                                        config.hidden_size)
        #     self.embedding_layer_1 = nn.Linear(config.hidden_size, 
        #                                        config.hidden_size)
        #     # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
        #     #                                    config.hidden_size)
        self.feedforward_layer = nn.Linear(config.hidden_size, 
                                           config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 
                                    config.num_labels)
        self.softmax = nn.Softmax(dim = -1)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.weighted_feature_gate = weighted_feature_gate
        if num_of_additional_features:
            self.num_of_binary_features = num_of_additional_features['num_of_binary_features']
            self.num_of_weighted_features = num_of_additional_features['num_of_weighted_features']
            
            if self.num_of_binary_features:
                self.embedding_layer = nn.Linear(self.num_of_binary_features, 
                                                   config.hidden_size)
                self.embedding_layer_1 = nn.Linear(config.hidden_size, 
                                                   config.hidden_size)
                # self.embedding_layer_2 = nn.Linear(config.hidden_size, 
                #                                    config.hidden_size)
            if self.num_of_weighted_features:
                self.WeightedFeatureLayerNorm = torch.nn.LayerNorm(config.hidden_size + self.num_of_weighted_features, eps=config.layer_norm_eps)
                # self.feedforward_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                #                                    config.hidden_size)
                self.weighted_feature_hidden_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                                                   config.hidden_size)
                self.weighted_feature_classifier = nn.Linear(config.hidden_size,
                                                            config.num_labels)
                if self.weighted_feature_gate:
                    self.gate_hidden_layer = nn.Linear(config.hidden_size + self.num_of_weighted_features, 
                                                       config.hidden_size)
                    self.gate_layer = nn.Linear(config.hidden_size, 1)
        else:
            self.num_of_binary_features = 0
            self.num_of_weighted_features = 0
        self.init_weights()
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.embeddings = self.roberta.embeddings
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        inputs_embeds += self.embeddings.token_type_embeddings(token_type_ids)
        
        #inputs_embeds += self.embeddings.position_embeddings(position_ids)
        binary_features = additional_features['binary_features']
        
        if self.num_of_binary_features:
            #inputs_embeds = torch.cat((inputs_embeds, additional_features), -1)
            #inputs_embeds += self.embedding_layer(additional_features)
            embedded_additional_feature = torch.tanh(self.embedding_layer(binary_features))
            # embedded_additional_feature = self.LayerNorm(embedded_additional_feature)
            embedded_additional_feature = self.dropout(embedded_additional_feature)
            inputs_embeds += torch.tanh(self.embedding_layer_1(embedded_additional_feature)) 
            
            
            
        outputs = self.roberta(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        #logits = self.classifier(output_hidden)
        
            
        output_hidden = self.feedforward_layer(sequence_output)
        
        
        
        output_hidden = torch.tanh(self.LayerNorm(output_hidden))
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden)
        emissions = logits
        boolean_attention_mask = attention_mask.type(torch.bool)
        if self.num_of_weighted_features:
            
            weighted_features = additional_features['weighted_features']
            weighted_features = self.WeightedFeatureLayerNorm(torch.cat((sequence_output, weighted_features), -1))
            weighted_feature_hidden = torch.tanh(self.LayerNorm(self.weighted_feature_hidden_layer(weighted_features)))
            weighted_feature_hidden = self.dropout(weighted_feature_hidden)
            weighted_feature_logits = self.weighted_feature_classifier(weighted_feature_hidden)
            #weighted_feature = torch.sigmoid(weighted_feature_logits)
            #weighted_feature = weighted_feature_logits

            if not self.weighted_feature_gate:
                emissions = emissions + weighted_feature_logits
            else:
                gate_input =  self.WeightedFeatureLayerNorm(weighted_features)
                gate_hidden = self.LayerNorm(self.gate_hidden_layer(gate_input))
                gate_hidden = self.dropout(torch.tanh(gate_hidden))
                weight_feature_gate = torch.sigmoid(self.gate_layer(gate_hidden))
                
                emissions = weight_feature_gate * emissions + (1- weight_feature_gate) * weighted_feature_logits
        
   
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(emissions, labels, boolean_attention_mask), self.crf.decode(emissions, boolean_attention_mask)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, boolean_attention_mask)
            return (None, sequence_of_tags)            
        
        

        
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
class EarlyFusionRobertaMLPCRF(RobertaPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_of_additional_features=0):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        try:
            self.num_of_additional_features = num_of_additional_features
        except:
            self.num_of_additional_features = 0
        
        self.num_of_additional_features = num_of_additional_features
        if self.num_of_additional_features:
            self.embedding_layer = nn.Linear(self.num_of_additional_features, 
                                               config.hidden_size)
        
        self.feedforward_layer = nn.Linear(config.hidden_size, 
                                           config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 
                                    config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        self.init_weights()

# [DOCS]    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features = None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.embeddings = self.roberta.embeddings
        inputs_embeds = self.embeddings.word_embeddings(input_ids)
        inputs_embeds += self.embeddings.token_type_embeddings(token_type_ids)
        
        #inputs_embeds += self.embeddings.position_embeddings(position_ids)
        if self.num_of_additional_features:
            #inputs_embeds = torch.cat((inputs_embeds, additional_features), -1)
            inputs_embeds += self.embedding_layer(additional_features)
            
            
        outputs = self.roberta(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        
        # print(sequence_output.shape)
        # output_hidden = sequence_output
        
        output_hidden = self.feedforward_layer(sequence_output)
        output_hidden = torch.tanh(output_hidden)
        output_hidden = self.dropout(output_hidden)
        
        logits = self.classifier(output_hidden + sequence_output)
        #logits = self.classifier(output_hidden)
        
        if labels is not None:
            negative_log_likelihood, sequence_of_tags = -1 * self.crf(logits, labels), self.crf.decode(logits)
            return negative_log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions)
            return (None, sequence_of_tags)
# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings."""

#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
#         self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

#     def forward(
#         self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
#     ):
#         if input_ids is not None:
#             input_shape = input_ids.size()
#         else:
#             input_shape = inputs_embeds.size()[:-1]

#         seq_length = input_shape[1]

#         if position_ids is None:
#             position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

#         if token_type_ids is None:
#             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)

#         embeddings = inputs_embeds + token_type_embeddings
#         if self.position_embedding_type == "absolute":
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings += position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings