import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = F.mish(self.layer1(x))
        x = F.mish(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
model     = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()
