import torch
import torch.nn as nn

alpha = 0.5 
lambda_reg = 0.01

S = nn.Sigmoid()
L = nn.BCELoss(reduction='mean')

def CustomLoss(f19, f34, mask):
    l1_loss = L(S(f34), mask)
    l2_loss = L(S(f19), mask)

    total_loss = l1_loss + alpha * l2_loss

    return total_loss

