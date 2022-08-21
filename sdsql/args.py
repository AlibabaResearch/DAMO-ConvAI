import torch
torch.cuda.set_device(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
