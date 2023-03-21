import numpy as np
import math

import torch
import torch.nn.functional as F

def gram_matrix(input):
    if input.dtype == torch.float16:
        input = input.to(torch.float32)
        flag = True
    a, b, c, d = input.size()  # a=batch size(=1)
    sqrt_sum = math.sqrt(a * b * c * d)  # for numerical stability
    features = input.view(a * b, c * d) / sqrt_sum  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    result = G
    if flag:
        return result.to(torch.float16)
    else:
        return result

def image_loss(source, target, args):
    if args.image_loss == 'semantic':
        source[-1] = source[-1] / source[-1].norm(dim=-1, keepdim=True)
        target[-1] = target[-1] / target[-1].norm(dim=-1, keepdim=True)
        return (source[-1] * target[-1]).sum(1)
    elif args.image_loss == 'style':
        weights = [1, 1, 1, 1, 1]
        loss = 0
        for cnt in range(5):
            loss += F.mse_loss(gram_matrix(source[cnt]), gram_matrix(target[cnt]))
        return -loss * 1e10 / sum(weights)

def text_loss(source, target, args):
    source_feat = source[-1] / source[-1].norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    return (source_feat * target).sum(1)
