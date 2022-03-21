import json
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import numpy as np
import pandas as pd
import re
from transformers import BertTokenizer, BertConfig
import unicodedata
from copy import deepcopy
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from neuralnet import *
from utils import *


def data_qc(paragrahp:str):
    paragrahp = unicodedata.normalize("NFKD", paragrahp)
    paragrahp = re.sub(r"(\(.*?\))", "", paragrahp)
    paragrahp = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", "", paragrahp)
    paragrahp = re.sub("'^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'", "", paragrahp)
    return paragrahp


def train_model(model, optimizer, scheduler, train_dataloader, valid_dataloader=None, epochs=2):
        loss_fct = torch.nn.MSELoss()
        
        for epoch in range(epochs):
            print(f"*****Epoch {epoch} Train Start*****")
            
            total_loss, batch_loss, batch_count = 0, 0, 0
        
            model.train()

            for step, batch in enumerate(train_dataloader):
                batch_count+=1
                
                batch = tuple(item.to(device) for item in batch)
            
                batch_input, batch_label = batch
                
                model.zero_grad()
            
                logits = model(**batch_input)

                loss = loss_fct(logits.view(-1), batch_label.view(-1))
                
                batch_loss += loss.item()
                total_loss += loss.item()
            
                loss.backward()
                
                clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                if (step % 10 == 0 and step != 0):
                    learning_rate = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step : {step}, LR : {learning_rate}, Avg Loss : {batch_loss / batch_count:.4f}")

                    batch_loss, batch_count = 0,0

            print(f"Epoch {epoch} Total Mean Loss : {total_loss/(step+1):.4f}")
            print(f"*****Epoch {epoch} Train Finish*****\n")
            
            if valid_dataloader:
                print(f"*****Epoch {epoch} Valid Start*****")
                valid_loss = validate(model, valid_dataloader, loss_fct)
                print(f"Epoch {epoch} Valid Loss : {valid_loss:.4f}")
                print(f"*****Epoch {epoch} Valid Finish*****\n")
            
            # save_checkpoint(".", model, optimizer, scheduler, epoch, total_loss/(step+1))

        print("Train Completed. End Program.")
        


def validate(model, valid_dataloader, loss_fct):
   
    model.eval()
    
    total_loss = 0
        
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(item.to(device) for item in batch)
            
        batch_input, batch_label = batch
            
        with torch.no_grad():
            logits = model(**batch_input)
            
        loss = loss_fct(logits.view(-1), batch_label.view(-1))
        total_loss += loss.item()
        
    total_loss = total_loss/(step+1)

    return total_loss


def initializer(optimizer, train_dataloader, epochs=2):
    """
    모델, 옵티마이저, 스케쥴러 초기화
    """

    optimizer = optimizer(
        model.parameters(),
        lr=2e-5,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * epochs
    print(f"Total train steps with {epochs+1} epochs: {total_steps}")

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    return optimizer, scheduler


if __name__ == '__main__':
    device = init_device()

    #### params ####
    batch_size = 32
    epochs = 2
    optimizer = AdamW
    #################

    # load data
    with open("./klue-sts-data/klue-sts-v1.1_train.json", "rt", encoding='utf8') as f:
        data = json.load(f)
    
    frame = np.full([len(data), 3], np.nan)
    df = pd.DataFrame(frame, columns=['sentence1', 'sentence2', 'label'])
    for idx, el in enumerate(data):
        df.loc[idx] = [el['sentence1'], 
                       el['sentence2'], 
                       el['labels']['real-label']]
    del data
    
    df[['sentence1', 'sentence2']] = df[['sentence1', 'sentence2']].applymap(data_qc)

    train_dataset, valid_dataset = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_dataset.iloc[:, :2].values.tolist(), 
                                  train_dataset['label'].tolist())
    valid_dataset = CustomDataset(valid_dataset.iloc[:, :2].values.tolist(), 
                                  valid_dataset['label'].tolist())
    del df

    # config model
    conf_info = Config(f"./config_files/config_subchar12367_bert.json")
    with open("./config_files/config_subchar12367_bert.json", "rt", encoding="utf8") as f:
        conf_subchar = json.load(f)

    cust_col_fn = cust_col_fn_init_tokenizer(
        tokenizer = BertTokenizer.from_pretrained(conf_info.tokenizer),
        path = conf_info.tokenizer)

    config = BertConfig(conf_info.bert)
    config.vocab_size = 12367

    model = BertSts(config = config,
                    add_fc = init_fclayer(config.hidden_size, 
                                          config.hidden_size,
                                          3, [512, 64], # 레이어 수, [뉴런 수]
                                          config.hidden_dropout_prob))
    weights = torch.load(conf_info.bert)

    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)

    weight_dict = deepcopy(model.state_dict())
    for name, weight in weights.items():
        if name in param_names:
            weight_dict[name] = weight

    load_weight_result = model.load_state_dict(weight_dict)
    print(load_weight_result)

    # init dataloader
    train_dataloader = DataLoader(train_dataset,
                                batch_size = batch_size,
                                sampler = RandomSampler(train_dataset),
                                collate_fn = cust_col_fn)

    valid_dataloader = DataLoader(valid_dataset,
                                batch_size = batch_size,
                                sampler = RandomSampler(valid_dataset),
                                collate_fn = cust_col_fn)

    # train
    model.to(device)
    optimizer, scheduler = initializer(optimizer, 
                                       train_dataloader, 
                                       epochs=epochs)
    train_model(model, 
                optimizer = optimizer,
                scheduler = scheduler,
                train_dataloader = train_dataloader, 
                valid_dataloader = valid_dataloader, 
                epochs = epochs)