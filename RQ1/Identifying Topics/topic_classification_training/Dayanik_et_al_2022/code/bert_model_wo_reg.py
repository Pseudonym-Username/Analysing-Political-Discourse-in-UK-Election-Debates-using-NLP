# from pytorch_transformers import *

#-----J0 2026 edit-------------------------------------------------------------
# from transformers import *
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.nn import BCEWithLogitsLoss
#------------------------------------------------------------------------------

import torch
import torch.nn as nn
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
import pdb


class TransformerModel(torch.nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        #JO 2026 edit---
#         self.bert = BertModel.from_pretrained(args['model_name'])
        self.bert = AutoModel.from_pretrained(args['model_name'])
        #----
        self.num_labels = args['num_labels']
        self.dropout = torch.nn.Dropout(args['dp'])

        self.use_knowledge = args['use_knowledge']
        if self.use_knowledge:
            self.V = nn.Linear(args['num_labels'], args['hs'])
        else:
            self.W = nn.Linear(args['hs'], args['num_labels'])
        
        #JO 2026 edit----
        self.contrastive_projection = nn.Sequential(
            nn.Linear(args['hs'], args['hs']),
            nn.ReLU(),
            nn.Dropout(0.1),
#             nn.BatchNorm1d(args['hs']),
            nn.Linear(args['hs'], 128)
        )
        #----
        
        # self.classifier = torch.nn.Linear(args['hs'], args['num_labels'])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, Smatrix=None):
        #JO 2026 edit----
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         pooled_output = self.dropout(pooled_output)
        
        outputs = self.bert(input_ids, 
#                             token_type_ids=token_type_ids, 
                            attention_mask=attention_mask)
        
#       pooled_output = outputs.pooler_output 
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:            
            pooled_output = outputs.last_hidden_state[:, 0, :]        
            
        pooled_output = self.dropout(pooled_output)                
        con_embeddings = self.contrastive_projection(pooled_output)
        con_embeddings = nn.functional.normalize(con_embeddings, dim=1)
        #----
        if Smatrix is not None:
            W_t = self.V(Smatrix)
            W = W_t.transpose(0, 1)
            logits = torch.matmul(pooled_output, W)
        else:
            logits = self.W(pooled_output)

        # logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            # pdb.set_trace()
            # JO 2026 edit----
#             return loss
            return loss, con_embeddings
            # ----
        else:
            #JO 2026 edit
            # return logits
            return logits, con_embeddings
            # ----
