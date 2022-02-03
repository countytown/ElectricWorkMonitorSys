import argparse
import numpy as np
import torch

from models.common import *

if __name__ == '__main__':

    weights_path = r'H:\Ubuntu训练\exp91\weights\\best.pt'
    is_half = True

    # Load pytorch model
    model = torch.load(weights_path, map_location=torch.device('cpu'))

    net = model['model']

    if is_half:
        net.half() # 把FP32转为FP16

    # print(model)

    ckpt = {'epoch': -1,
            'best_fitness': model['best_fitness'],
            'training_results': None,
            'model': net,
            'optimizer': None}

    # Save .pt
    torch.save(ckpt, r'H:\Ubuntu训练\exp91\weights\\cuted_best.pt')
    # for name, parameters in model.named_parameters():
    #     # print(name,':',parameters.size())
    #     print(parameters.dtype)
