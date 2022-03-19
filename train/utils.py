from typing import Union
import json
from pathlib import Path
import re
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class Config:
    """Config class"""

    def __init__(self, json_path_or_dict: Union[str, dict]) -> None:
        """Instantiating Config class
        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    def save(self, json_path: Union[str, Path]) -> None:
        """Saving config to json_path
        Args:
            json_path (Union[str, Path]): filepath of config
        """
        with open(json_path, mode="w") as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path_or_dict) -> None:
        """Updating Config instance
        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    @property
    def dict(self) -> dict:
        return self.__dict__


def data_qc(paragrahp:str):
    paragrahp = re.sub(r"(\(.*?\))", "", paragrahp)
    paragrahp = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", "", paragrahp)
    paragrahp = re.sub("'^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'", "", paragrahp)
    return paragrahp


def cust_col_fn_init_tokenizer(tokenizer, path):
    tokenizer = BertTokenizer.from_pretrained(f"{path}")

    def custom_collate_fn(batch):
        nonlocal tokenizer

        input_list, target_list = [], []
        
        for _input, _target in batch:
            input_list.append(_input)
            target_list.append(_target)
        
        tensorized_input = tokenizer(
            input_list,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        tensorized_label = torch.tensor(target_list, dtype = torch.float)
        
        return tensorized_input, tensorized_label

    return custom_collate_fn


class CustomDataset(Dataset):
    """
    - input_data: list of string
    - target_data: list of int
    """

    def __init__(self, input_data:list, target_data:list) -> None:
        self.X = input_data
        self.Y = target_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        return X, Y


def init_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")

    else:
        device = torch.device("cpu")
    print(device)

    return device

def save_checkpoint(path, model, optimizer, scheduler, epoch, loss):
    file_name = f'{path}/model.ckpt.{epoch}'
    
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss' : loss
        }, 
        file_name
    )
    
    print(f"Saving epoch {epoch} checkpoint at {file_name}")