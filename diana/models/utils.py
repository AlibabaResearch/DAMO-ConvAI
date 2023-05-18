import torch.nn.functional as F
from torch import nn
import torch
import copy
#adaptive decision boundary
#[code]https://github.com/thuiar/Adaptive-Decision-Boundary
#[paper]https://www.aaai.org/AAAI21Papers/AAAI-9723.ZhangH.pdf
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_metric(a,b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = (a*b).sum(dim=2)
 #   logits = -logits+1
    return logits
   

class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim

        self.delta = nn.Parameter(torch.ones(20).cuda())
        nn.init.normal_(self.delta)

        
    def forward(self, pooled_output, centroids_, labels,group=0):
        centroids = copy.deepcopy(centroids_.detach())
        logits = cosine_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1) 
        delta = F.softplus(self.delta)
        #c = torch.Tensor([[0.0]*768]*(centroids[labels]).size(0)).to(pooled_output.device)#
      #  print(self.delta.size())
      #  print("labels",labels.size())
        c = centroids[labels]
        c = c/torch.norm(c, p=2, dim=1, keepdim=True)
       # c.requires_grad = False
        d = delta[labels]
        x = pooled_output
        
        euc_dis = 1-((c*x).sum(dim=1))#torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()

        
        return loss, delta 