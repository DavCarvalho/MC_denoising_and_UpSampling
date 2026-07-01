import torch
from mc_upsampling import Upsampler, visualizar  

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = Upsampler().to(dev)
net.load_state_dict(torch.load("upsampler_div2k_v2.pt", map_location=dev))
net.eval()

visualizar(net, dev)