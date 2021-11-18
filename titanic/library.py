!pip install kaggle
from google.colab import files
files.upload()
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
#elu relu sigmoid mish tanh selu hardtanh s
