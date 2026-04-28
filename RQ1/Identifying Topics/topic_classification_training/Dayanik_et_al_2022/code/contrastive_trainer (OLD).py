import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_model import TransformerModel  
from bert_data import * 
import numpy as np

class HierarchyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(HierarchyContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels, distance_matrix):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalise embeddings to unit sphere for cosine similarity       
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # calculate cosine similarity between all pairs in the batch
        logits = torch.div(
            torch.matmul(embeddings, embeddings.T),
            self.temperature
        )
        # Subtract max logit for stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Calculate average tree dist between the labels of two samples
        label_overlap = torch.matmul(labels.float(), labels.float().T)
        # Calculate hieraarchical similarity
        max_dist = distance_matrix.max()
        hierarchy_sim = (max_dist - distance_matrix) / max_dist
        # map samples to their average hierarchical similarity
        mask = torch.matmul(labels.float(), torch.matmul(hierarchy_sim, labels.float().T))
        mask = mask / (label_overlap + 1e-8)
        mask = F.normalize(mask, p=1, dim=1)
        
        # log soft-max over the similarities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # final loss = mean of masked log probabilities
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos
        
        return loss.mean()
    
def get_tree_distance_matrix(label_list):
    # 0 - same label, 1 - sibling, 2 - different branches
    
    num_labels = len(label_list)
    dist_mat = np.zeros((num_labels, num_labels))
    
    for i, lab1 in enumerate(label_list):
        for j, lab2 in enumerate(label_list):
            if lab1 == lab2:
                dist_mat[i, j] = 0
            else:
                p1 = int(lab1) // 100
                p2 = int(lab2) // 100
                if p1 == p2:
                    dist_mat[i, j] = 1 
                else:
                    dist_mat[i, j] = 2 
                    
    return torch.tensor(dist_mat).float()