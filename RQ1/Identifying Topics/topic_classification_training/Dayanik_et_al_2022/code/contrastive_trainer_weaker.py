import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchySupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels, distance_matrix):        
        device = embeddings.device
        B = embeddings.shape[0]

        
        embeddings = F.normalize(embeddings, dim=1)

        
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        
        logits_mask = torch.ones_like(sim) - torch.eye(B, device=device)

        hierarchy_sim = torch.exp(-distance_matrix)  # [L, L]

        label_sim = labels.float() @ hierarchy_sim @ labels.float().T  # [B, B]

        intersection = labels @ labels.T
        union = labels.sum(1, keepdim=True) + labels.sum(1) - intersection
        jaccard = intersection / (union + 1e-8)

        sim_targets = 0.5 * jaccard + 0.5 * (label_sim / (label_sim.max() + 1e-8))

        pos_mask = (sim_targets > 0.3).float() * logits_mask

        pos_weight = sim_targets * pos_mask
        pos_weight = pos_weight / (pos_weight.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = F.log_softmax(sim, dim=1)

        loss = -(pos_weight * log_prob).sum(dim=1)

        loss = loss * (self.temperature / self.base_temperature)

        return loss.mean()
    
def get_tree_distance_matrix(label_list):    
    num_labels = len(label_list)
    dist_mat = np.zeros((num_labels, num_labels))
    
    for i, lab1 in enumerate(label_list):
        l1 = int(lab1)
        for j, lab2 in enumerate(label_list):
            l2 = int(lab2)
            
            if l1 == l2:
                dist_mat[i, j] = 0.0
            
            else:
                p1, p2 = l1 // 100, l2 // 100
                
                if p1 == p2:
                    if l1 % 100 == 0 or l2 % 100 == 0:
                        dist_mat[i, j] = 1.0   # parent-child
                    else:
                        dist_mat[i, j] = 2.0   # siblings
                else:
                    dist_mat[i, j] = 4.0       # different branches
    
    return torch.tensor(dist_mat).float()