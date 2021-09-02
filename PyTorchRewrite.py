# import os
import torch
# import pandas as pd
import gzip
import pickle
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from pathlib import Path
from torch import optim
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)



with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
#Loading the Data into Dataloader
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid,y_valid)

# class NeuralNetwork(nn.Module):
#     """Implement NeuralNetwork."""

#     def __init__(self,bias=True):
#         """
#         Initialize NeuralNetwork.

#         Bias is included
#         """
#         super().__init__()
#         self.lin1=nn.Linear(784,30,bias=bias)
#         self.lin2=nn.Linear(30,100,bias=bias)
#         self.lin3=nn.Linear(100,10,bias=bias)


#     def forward(self, xb):
#         """Implement feed forward."""
#         sigmoid=nn.Sigmoid()
#         tanh=nn.Tanh()
#         xb=self.lin1(xb)
#         xb=tanh(xb)
#         xb=self.lin2(xb)
#         xb=tanh(xb)
#         xb=self.lin3(xb)
#         xb=tanh(xb)
#         return xb

class NeuralNetwork(nn.Module):
    """Implement NeuralNetwork."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        #self.avgpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)
        self.avgpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.lin1 = nn.Linear(18 * 14 * 14,100)
        self.lin2 = nn.Linear(100,10)

    def forward(self, xb):

        xb = xb.view(-1, 1, 28, 28)
        #print(xb.size())
        xb = F.relu(self.conv1(xb))
        #print(xb.size())
        #xb = F.relu(self.avgpool1(xb))
        xb = F.relu(self.conv2(xb))
        #print(xb.size())
        #xb = F.relu(self.conv3(xb))
        #xb = F.avg_pool2d(xb, 4)
        xb = F.relu(self.avgpool2(xb))
        #print(xb.size())

        #Linear layers
        xb = xb.view(xb.size(0), -1)
        #print(xb.size())
        xb = F.relu(self.lin1(xb))
        #print(xb.size())
        xb = F.relu(self.lin2(xb))
        #print(xb.size())
        return xb.view(-1, xb.size(1))


def getmodel(lr):
    """Call model neuralNetwork and optimizer."""
    model=NeuralNetwork()
    return model, optim.SGD(model.parameters(),lr=lr)


def nll(input,target):
    """Implement negative log likelihood."""
    return -input[range(target.shape[0]),target].mean()

cross_entropy=nn.CrossEntropyLoss()



def fit(epochs,loss_func=cross_entropy,bs=64,lr=0.5):
    #Loading Data
    train_dl = DataLoader(train_ds, batch_size=bs)
    valid_dl = DataLoader(valid_ds,batch_size=bs*2)
    #Getting Model
    model,opt=getmodel(lr=lr) #Getting the model and its optimizer.
    model.to(device)
    for epoch in range(epochs):
        model.train()
        #training
        for xb,yb in train_dl:

            xb,yb=xb.to(device),yb.to(device)

            pred=model(xb)
            loss=loss_func(pred,yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
        #evaluating
        model.eval()
        with torch.no_grad():
            valid_loss,correct=0,0
            for xb,yb in valid_dl:
                xb,yb=xb.to(device),yb.to(device)
                pred=model(xb)

                correct += (pred.argmax(1) == yb).type(torch.float).sum().item()
                #valid_loss += loss_func(pred,yb)
            
        
        #print(f"Epoch Number: {epoch} "+"\t"+f"{valid_loss/len(valid_dl)}")
        print(f"Epoch Number: {epoch} "+"\t"+f"{correct}/{y_valid.shape[0]}")




