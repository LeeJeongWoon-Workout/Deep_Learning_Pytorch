from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import torchvision

drive.mount('/content/drive')
!unzip '/content/drive/MyDrive/Colab Notebooks/deeplearning/deeplearning2021competition.zip'

