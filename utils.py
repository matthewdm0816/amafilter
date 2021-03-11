import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric as tg

def transform(samplePoints=2500, k=32):
    def f(data):
        data = tg.transforms.NormalizeScale()(data) # normalize to [-1, 1]
        data = tg.transforms.SamplePoints(samplePoints)(data)
        # data = tg.transforms.KNNGraph(k=k)(data)
        return data
    return f

def check_dir(path, color=None):
    """
    check directory if avaliable
    """
    import os, colorama
    if not os.path.exists(path):
        print("" if color is None else color + "Creating path %s" % path)
        os.makedirs(path, exist_ok=True)

def add_noise(pc, scale=0.01):
    """
    add gaussian noise
    """
    noise = torch.randn(pc.shape).to(pc) * scale
    return pc + noise

def add_multiplier_noise(pc, multiplier=10):
    """
    add gaussian noise according to standard deviation
    """
    std = torch.std(pc)
    # print(std)
    return pc + torch.randn(pc.shape).to(pc) * std * multiplier

def color_jitter(pos, color, scale=0.01):
    """
    wrapped jitterer on pc(pos + color)
    """
    color = add_noise(color)
    return torch.cat(pos, color, dim=-1) # concat on last dim

def mse(fake, real):
    return F.mse_loss(fake, real, reduction='mean')

def psnr(fake, real, max=1.):
    # fake, real: N * FIN
    return mse_to_psnr(mse(fake, real), max=max)

def mse_to_psnr(mse, max=1.):
    return 10 * torch.log10(max ** 2 / mse)
    
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def load_model(model, optimizer, f: str, optim: str, e: int, evaluate=None):
    model.load_state_dict(torch.load(f))
    print("Loaded milestone with epoch %d at %s" % (e, f))
    if optim is not None:
        optimizer.load_state_dict(torch.load(optim))
        print("Loaded milestone optimizer with epoch %d at %s" % (e, optim))
    if evaluate is not None:
        evaluate(model)

class layers():
    def __init__(self, ns):
        self.ns = ns
        self.iter1 = iter(ns)
        self.iter2 = iter(ns) # iterator of latter element
        next(self.iter2)

    def __iter__(self):
        return self

    def __next__(self):
        return (next(self.iter1), 
                next(self.iter2))


def module_wrapper(f):
    class module(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return f(x)
    
    return module()

def parallel_cuda(batchs, device):
    batchs = [data.to(device) for data in batchs]
    reals = [data.pos.to(device) for data in batchs]
    # jittered = [ for real in reals]
    labels = [data.y for data in batchs] # actually batchs
    labels = torch.cat(labels).to(device)
    bs = labels.shape[0] 
    return batchs, reals, labels

def tensorinfo(t):
    return "%f, %f, %f" % (t.max().item(), t.median().item(), t.min().item())
