import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(HierarchyContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels, distance_matrix):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # normalise embeddings cosine similarity
        embeddings = F.normalize(embeddings, dim=1)
        
        # pairwise similarity
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # remove self-comparisons
        logits_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=device)
        sim_matrix = sim_matrix * logits_mask
        
        # build hierarchical similarity between samples
        max_dist = distance_matrix.max()
        hierarchy_sim = (max_dist - distance_matrix) / max_dist  
        
        # map samples to their average hierarchical similarity
        mask = torch.matmul(labels.float(), torch.matmul(hierarchy_sim, labels.float().T))
#         target_distribution = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        # changes after best run
        # sample-to-sample similarity via labels
        sample_sim = labels.float() @ hierarchy_sim @ labels.float().T
        
        # only treat pairs with meaningful overlap as positives
        pos_mask = (sample_sim > 0).float() * logits_mask
        # only same labels are positive
#         exact_match = (labels @ labels.T) > 0
#         pos_mask = exact_match.float() * logits_mask
        
        # weight positives by hierarchy strength
#         pos_weight = sample_sim * pos_mask
        # increase the soft influence of the hierarchy
#         pos_weight = pos_mask + 0.3 * sample_sim
#         pos_weight = pos_weight * pos_mask
    
        pos_weight = sample_sim * pos_mask
        
        # normalise weights per anchor (avoid dominance by multi-label samples)        
        pos_weight = pos_weight / (pos_weight.sum(dim=1, keepdim=True) + 1e-8)
        label_count = labels.sum(dim=1, keepdim=True)  # number of labels per sample
        pos_weight = pos_weight / (label_count + 1e-8)
        
        #add hard negatives
#         neg_mask = (sample_sim == 0).float()
        neg_mask = (sample_sim < 0.1).float()
        
        # log-softmax over all pairs
        log_prob = F.log_softmax(sim_matrix, dim=1)
        # compute loss over positives
        loss_pos = -(pos_weight * log_prob).sum(dim=1)
        # compute neg loss
        loss_neg = -0.1 * (neg_mask * log_prob).sum(dim=1)
        
        loss = loss_pos + loss_neg
        
        return loss.mean()
    
def get_tree_distance_matrix(label_list):
    # 0 - same label, 0.5-parent-child, 1 - sibling, 3 - different branches
    
    num_labels = len(label_list)
    dist_mat = np.zeros((num_labels, num_labels))
    
    for i, lab1 in enumerate(label_list):
        l1 = int(lab1)
        for j, lab2 in enumerate(label_list):
            l2 = int(lab2)
            if l1 == l2:
                dist_mat[i, j] = 0
            else:
                p1, p2 = l1 // 100, l2 // 100
                if p1 == p2:                    
                    if l1 % 100 == 0 or l2 % 100 == 0:
                        dist_mat[i, j] = 0.5  # parent-child
                    else:
                        dist_mat[i, j] = 1.0  # siblings
                else:
                    dist_mat[i, j] = 3.0 # completely different
                    
    return torch.tensor(dist_mat).float()