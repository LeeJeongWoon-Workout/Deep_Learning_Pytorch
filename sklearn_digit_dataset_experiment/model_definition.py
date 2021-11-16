import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
#elu relu sigmoid mish tanh selu hardtanh step

class Model1(nn.Module):
    def __init__(self, input_dim):
        super(Model1, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.elu(self.layer1(x))
        x = F.elu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class Model2(nn.Module):
    def __init__(self, input_dim):
        super(Model2, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


class Model3(nn.Module):
    def __init__(self, input_dim):
        super(Model3, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class Model4(nn.Module):
    def __init__(self, input_dim):
        super(Model4, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.mish(self.layer1(x))
        x = F.mish(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class Model5(nn.Module):

   def __init__(self, input_dim):
        super(Model5, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
   def forward(self, x):
        x = F.hardtanh(self.layer1(x))
        x = F.hardtanh(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class Model6(nn.Module):
    def __init__(self, input_dim):
        super(Model6, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


class Model7(nn.Module):
    def __init__(self, input_dim):
        super(Model7, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

class Model8(nn.Module):
    def __init__(self, input_dim):
        super(Model8, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
