from typing import Union
from pathlib import Path
import hashlib
from _hashlib import HASH as Hash
import random
import numpy as np
import torch
import os

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# to be called my md5_dir
def md5_update_from_dir(directory: Union[str, Path], hash: Hash) -> Hash:
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            hash = md5_update_from_file(path, hash)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash

# given a directory of data, return the hash
def md5_dir(directory: Union[str, Path]) -> str:
    return str(md5_update_from_dir(directory, hashlib.md5()).hexdigest())

def md5_update_from_file(filename: Union[str, Path], hash: Hash) -> Hash:
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

# check if parameters are all the same
def check_two_models_have_same_weights(model1, model2):
    # Get state dictionaries of the models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if keys are the same
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    # Check if values are the same for each key
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True
