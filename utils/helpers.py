# Ben Kabongo
# September 2025


import numpy as np
import random
import torch


def pprint(config, log: str) -> None:
    if hasattr(config, "logger"):
        config.logger.info(log)
    else:
        print(log)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)