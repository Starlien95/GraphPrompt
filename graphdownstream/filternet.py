import torch
import torch.nn as nn
import torch.nn.functional as F

# class MaxGatedFilterNet(nn.Module):
#     def __init__(self, pattern_dim, graph_dim):
#         super(MaxGatedFilterNet, self).__init__()
#         self.g_layer = nn.Linear(graph_dim, pattern_dim)
#         self.f_layer = nn.Linear(pattern_dim, 1)

#         # init
#         scale = (1/pattern_dim)**0.5
#         nn.init.normal_(self.g_layer.weight, 0.0, scale)
#         nn.init.zeros_(self.g_layer.bias)
#         nn.init.normal_(self.f_layer.weight, 0.0, scale)
#         nn.init.ones_(self.f_layer.bias)
    
#     def forward(self, p_x, g_x):
#         max_x = torch.max(p_x, dim=1, keepdim=True)[0].float()
#         g_x = self.g_layer(g_x.float())
#         f = self.f_layer(g_x * max_x)
#         return F.sigmoid(f)

class MaxGatedFilterNet(nn.Module):
    def __init__(self):
        super(MaxGatedFilterNet, self).__init__()
    
    def forward(self, p_x, g_x):
        max_x = torch.max(p_x, dim=1, keepdim=True)[0]
        if max_x.dim() == 2:
            return g_x <= max_x
        else:
            return (g_x <= max_x).all(keepdim=True, dim=2)


