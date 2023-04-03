import torch
import numpy as np

def set_seed(seed):
    """ Fix training conditions """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(seed)