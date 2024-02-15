import torch
import numpy as np


def auto_numpy(method):
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = [], {}
        for a in list(args):
            if torch.is_tensor(a):
                new_args.append(a.detach().cpu().numpy())
            elif isinstance(a, list):
                new_a = [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in a]
                new_args.append(new_a)
            else:
                new_args.append(a)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                new_kwargs[key] = value.detach().cpu().numpy()
            else:
                new_kwargs[key] = value
        return method(*new_args, **new_kwargs)

    return wrapper


def auto_tensor(method):
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = [], {}
        for a in list(args):
            if isinstance(a, np.ndarray):
                new_args.append(torch.tensor(a))
            else:
                new_args.append(a)
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                new_kwargs[key] = torch.tensor(value)
            else:
                new_kwargs[key] = value
        return method(*new_args, **new_kwargs)

    return wrapper
