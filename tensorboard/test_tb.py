import torch
import numpy as np
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()
    for n in range(100):
        loss_test = torch.rand(1)
        writer.add_scalar('Loss', loss_test, n)
    writer.close()

