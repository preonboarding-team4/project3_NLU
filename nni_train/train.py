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
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import nni

from neuralnet import *
from utils import *


def data_qc(paragrahp:str):
    paragrahp = unicodedata.normalize("NFKD", paragrahp)
    paragrahp = re.sub(r"(\(.*?\))", "", paragrahp)
    paragrahp = re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", "", paragrahp)
    paragrahp = re.sub("'^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'", "", paragrahp)
    return paragrahp


def train_model(model, params, optimizer, scheduler, train_dataloader, valid_dataloader=None, epochs=2, patience=5):
        stats_name = ["MSE loss", "f1 score", "pearson r"]
        digit = len(str(len(train_dataloader)))
        early_stopping = EarlyStopping(patience = patience)

        best_loss = np.inf
        loss_fct = torch.nn.MSELoss()
        for epoch in range(1, epochs+1):
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
                
                if (step % 25 == 0 and step != 0):
                    learning_rate = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step: {step:{digit}d}, LR: {learning_rate:.2e}, Avg Loss: {batch_loss / batch_count:.4f}")

                    batch_loss, batch_count = 0,0
                nni.report_intermediate_result(batch_loss)

            print(f"Epoch {epoch} Total Mean Loss : {total_loss/(step+1):.4f}")
            print(f"*****Epoch {epoch} Train Finish*****\n")
            
            if valid_dataloader:
                print(f"*****Epoch {epoch} Valid Start*****")
                valid_result = validate(model, valid_dataloader, loss_fct)
                valid_result = {name:res for name, res in zip(stats_name, valid_result)}

                for name in valid_result:
                    print(f"Epoch {epoch} {name} Loss : {valid_result[name]:.4f}")
                print(f"*****Epoch {epoch} Valid Finish*****\n")

            if valid_result['MSE loss'] < best_loss:
                best_loss = valid_result['MSE loss']

            save_checkpoint(model, valid_result, params)
            early_stopping(valid_result['MSE loss'])

            if early_stopping.early_stop:
                print('terminating because of early stopping.')
                break

        nni.report_final_result(best_loss)
        print("Train Completed. End Program.")
        

def validate(model, valid_dataloader, loss_fct):
   
    model.eval()
    
    total_loss = 0
    Logit, Batch_label = [], []
        
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(item.to(device) for item in batch)
            
        batch_input, batch_label = batch
            
        with torch.no_grad():
            logits = model(**batch_input)
            
        loss = loss_fct(logits.view(-1), batch_label.view(-1))
        total_loss += loss.item()

        Logit.append(logits.view(-1))
        Batch_label.append(batch_label.view(-1))

    total_loss = total_loss/(step+1)

    Logit = torch.concat(Logit).cpu().numpy()
    Batch_label = torch.concat(Batch_label).cpu().numpy()

    Logit = (Logit >= 3).astype(np.int8)
    Batch_label = (Batch_label >= 3).astype(np.int8)
    
    f1_score_ = f1_score(Batch_label, Logit)
    pearsonr_ = pearsonr(Batch_label, Logit)[0]
    return total_loss, f1_score_, pearsonr_


def initializer(optimizer, train_dataloader, lr=2e-5, epochs=2):
    """
    모델, 옵티마이저, 스케쥴러 초기화
    """

    optimizer = optimizer(
        model.parameters(),
        lr=lr,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * epochs
    print(f"Total train steps with {epochs} epochs: {total_steps}")

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps = total_steps,
        num_cycles = 1
    )

    return optimizer, scheduler


if __name__ == '__main__':
    device = init_device()

    # get params from nni
    params = nni.get_next_parameter()

    #### params ####
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    optimizer = getattr(torch.optim, params['optimizer'])
    #################
    epochs = 30

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

    if params["num_layer"] > 0:
        model = BertSts(config = config,
                        add_fc = init_fclayer(config.hidden_size, 
                                            config.hidden_size,
                                            params["num_layer"], 
                                            [params["neurons"] for _ in range(int(params["num_layer"])-1)], 
                                            config.hidden_dropout_prob))
    else:
        model = BertSts(config = config)
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
                                       lr=learning_rate,
                                       epochs=epochs)
    train_model(model = model, 
                params = params,
                optimizer = optimizer,
                scheduler = scheduler,
                train_dataloader = train_dataloader, 
                valid_dataloader = valid_dataloader, 
                epochs = epochs,
                patience = 5)