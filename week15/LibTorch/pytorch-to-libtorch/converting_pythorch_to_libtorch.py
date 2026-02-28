# Let's import the necessary libraries.
import torch
import torch.nn as nn
device = "cpu" if not torch.cuda.is_available() else "cuda"

# create a network in Pytorch
class LeNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature = nn.Sequential(
        #input (28,28,1), #output (24,24,6) formula [(28-5)/1]+1=24
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5), 
        nn.ReLU(),
         #input (24,24,6), output (12,12,6)
        nn.MaxPool2d(kernel_size=2),
        #input (12,12,6) output (8,8,16) formula [(12-5)]+1 = 8
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5), 
        nn.ReLU(),
        #input (8,8,16) output (4,4,16)
        nn.MaxPool2d(kernel_size=2) 
    )
    self.classifier = nn.Sequential(
        nn.Linear(16*4*4,512), #input(4*4*16) output 512
        nn.ReLU(),

        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )
  def forward(self,x):
    x = self.feature(x)
    x = x.view(x.shape[0],-1)
    x = self.classifier(x)
    return x

Model = LeNet().to(device)

# create an IR of the `LeNet` class and save it as `LeNet.pt`.
script_model = torch.jit.script(Model)
torch.jit.save(script_model, "./LeNet.pt")

