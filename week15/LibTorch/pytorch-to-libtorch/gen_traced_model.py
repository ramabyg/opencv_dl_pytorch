import torch
import torch.nn as nn

device = "cpu" if not torch.cuda.is_available() else "cuda"

class LeNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature = nn.Sequential(
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) 
    )
    self.classifier = nn.Sequential(
        nn.Linear(16*4*4,512),
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
Model.eval()

print("Creating traced model...")
example_input = torch.randn(1, 1, 28, 28).to(device)
traced_model = torch.jit.trace(Model, example_input)

print("Saving traced model...")
torch.jit.save(traced_model, "./LeNet.pt")
print("Model saved successfully!")

print("Testing loaded model...")
loaded_model = torch.jit.load("./LeNet.pt")
test_output = loaded_model(example_input)
print(f"Test output shape: {test_output.shape}")
print("SUCCESS!")
