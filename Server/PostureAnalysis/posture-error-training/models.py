"""Temporal multi-label classifiers; sigmoid outputs independent error probabilities."""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self,features=69,outputs=2):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(features,48,5,padding=2),nn.ReLU(),nn.Conv1d(48,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(64,outputs))
    def forward(self,x):return self.net(x.transpose(1,2))

class GRU(nn.Module):
    def __init__(self,features=69,outputs=2):
        super().__init__();self.gru=nn.GRU(features,48,batch_first=True);self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(48,outputs))
    def forward(self,x):
        out,_=self.gru(x)
        return self.head(out.mean(dim=1))

MODELS={'cnn':CNN,'gru':GRU}
