import torch
import torch.nn as nn

alpha = 0.5 
lambda_reg = 0.01

S = nn.Sigmoid()
L = nn.BCELoss(reduction='mean')

def CustomLoss(model_output, mask):
    l1_loss = L(S(model_output), mask)

    total_loss = l1_loss

    return total_loss


