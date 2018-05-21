from PIL import Image
import numpy as np
import torch
from torch import nn


def eps_decay(step, num_steps, eps_start, eps_end):
    eps = eps_start - (eps_start - eps_end) * min(1, step / (0.7 * num_steps))
    return eps

def pongScreenCut(img):
    img = img[34:194, :, :]
    img = np.array(Image.fromarray(img).convert(mode='L'))
    state = torch.from_numpy(img)
    state = state.unsqueeze(0).float().unsqueeze(0)
    return state

def get_total_loss(agent, batch_size):
    k = int(len(agent.memory)/batch_size)
    total_loss = 0
    for _ in range(k):
        loss = agent.get_loss(batch_size)
        total_loss += loss.data.cpu().numpy()
    return total_loss


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)
