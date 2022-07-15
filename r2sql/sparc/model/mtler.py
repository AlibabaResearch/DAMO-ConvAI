import torch
import torch.nn as nn


class Mtler(nn.Module):
    def __init__(self, task_num=2):
        super(Mtler, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        
    def forward(self, losses):
        flag = True
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            if flag:
                re_loss = precision * loss + self.log_vars[i]
            else:
                re_loss += precision * loss + self.log_vars[i]
        return re_loss, self.log_vars
        
          
            
            
