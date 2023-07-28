
import os
import random
import itertools
import time
import json
import copy
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, logging, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


logging.set_verbosity_error()


class SentenceDataset(Dataset):

    def __init__(self, df, df_type, maxlen, llm_model, balanced):
        self.maxlen = maxlen
        self.df_type = df_type
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model) 

        if self.df_type in [DOPPIA, Q_DOPPIA]:
            df = df.drop_duplicates(subset=['tAnswer']).reset_index(drop=True)

        if balanced:
            zero_data = df[df.label == 0].sample(n=min(len(df[df.label == 1]), len(df[df.label == 0])), random_state=SEED).reset_index(drop=True)
            one_data = df[df.label == 1].sample(n=min(len(df[df.label == 1]), len(df[df.label == 0])), random_state=SEED).reset_index(drop=True)

            self.df = pd.concat([zero_data, one_data]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        else:
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        if self.df_type == DOPPIA:
            sent  = str(self.df.loc[index, 'tAnswer']) + ' [SEP] ' 
            sent += str(self.df.loc[index, 'sDefaultAnswer'])

        elif self.df_type == CONCETTI:
            sent  = str(self.df.loc[index, 'tAnswer']) + ' [SEP] ' 
            sent += str(self.df.loc[index, 'sConcept'])

        elif self.df_type == Q_DOPPIA:
            sent  = str(self.df.loc[index, 'sQuestion']) + ' [SEP] ' 
            sent += str(self.df.loc[index, 'tAnswer']) + ' [SEP] ' 
            sent += str(self.df.loc[index, 'sDefaultAnswer'])

        elif self.df_type == Q_CONCETTI:
            sent  = str(self.df.loc[index, 'sQuestion']) + ' [SEP] ' 
            sent += str(self.df.loc[index, 'tAnswer']) + ' [SEP] '
            sent += str(self.df.loc[index, 'sConcept'])


        encoded = self.tokenizer(sent,
                                 padding='max_length',  
                                 truncation=True,       
                                 max_length=self.maxlen,
                                 return_tensors='pt')   
        
        token_ids = encoded['input_ids'].squeeze(0)           
        attn_masks = encoded['attention_mask'].squeeze(0)
        label = torch.tensor(self.df.loc[index, 'label'])

        return token_ids, attn_masks, label
    
    def get_labels(self):
        return self.df.label.tolist()
    
    def get_answerids(self):
        return self.df.iAnswerId.tolist()
    
    def get_concweigths(self):
        return self.df.iPeso.tolist()


class Classifier(nn.Module):

    def __init__(self, cls_type, cls_params):
        super(Classifier, self).__init__()

        self.cls_type = cls_type
        self.cls_params = cls_params

        self.llm_layer = AutoModel.from_pretrained(self.cls_params['llm'])

        if self.cls_params['freeze']:
            for param in self.llm_layer.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=cls_params['dropout_rate'])
        self.cls_layer = nn.Linear(self.llm_layer.config.hidden_size, 1)


    @autocast()
    def forward(self, input_ids, attn_masks):

        if self.cls_params['llm'] == ELECTRA:
            last_hidden_states = self.llm_layer(input_ids, attn_masks, return_dict=False)[0]
        else:
            last_hidden_states, pooler_output = self.llm_layer(input_ids, attn_masks, return_dict=False)

        if self.cls_type == FF:

            if self.cls_params['emb_type'] == POOLER:
                emb = pooler_output
            elif self.cls_params['emb_type'] == CLS:
                emb = last_hidden_states[:, 0, :]
            elif self.cls_params['emb_type'] == EMB_MEAN:
                emb = last_hidden_states.mean(dim=1)
            elif self.cls_params['emb_type'] == EMB_MAX:
                emb, _ = last_hidden_states.max(dim=1)
            elif self.cls_params['emb_type'] == EMB_MIN:
                emb, _ = last_hidden_states.min(dim=1)

            logits = self.cls_layer(self.dropout(emb))

        return logits
    

    def train_cls(self, train_data, val_data, epochs, criterion, optimizer, scaler, scheduler, device, timestamp):

        train_losses = []  
        val_losses = []  
        f1_scores = [] 
        best_f1 = 0.0
        best_ep = 0
        best_model = copy.deepcopy(self.state_dict())

        for ep in range(epochs):
            self.train()
            ep_loss = 0.0

            for batch in train_data:
                optimizer.zero_grad()
                input_ids, attn_masks, labels = [b.to(device) for b in batch]

                with autocast():
                    logits = self(input_ids, attn_masks)
                    loss = criterion(logits.squeeze(-1), labels.float())

                ep_loss += loss.item()

                # Backpropagation
                scaler.scale(loss).backward()

                # Optimizer step
                old_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if(scaler.get_scale() >= old_scale):
                    scheduler.step()

            # Evaluation

            val_loss, f1score = self.eval_model(val_data, criterion, device)
            f1_scores.append(f1score)
            val_losses.append(val_loss)

            if f1score > best_f1:
                best_f1 = f1score
                best_model = copy.deepcopy(self.state_dict())
                best_ep = ep+1

            train_losses.append(ep_loss/len(train_data))

        torch.save(best_model, 'results/models/' + timestamp + '.pth')

        return train_losses, f1_scores, val_losses, best_ep


    def eval_model(self, val_data, criterion, device):
        self.eval()

        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_data:
                input_ids, attn_masks, labels = [b.to(device) for b in batch]

                with autocast():
                    logits = self(input_ids, attn_masks)
                    loss = criterion(logits.squeeze(-1), labels.float())
                    val_loss += loss.item()

                    val_preds.extend([1 if p >= 0.5 else 0 for p in torch.sigmoid(logits.squeeze(-1)).cpu().numpy()])
                    val_labels.extend(labels.cpu().numpy())

        return val_loss/len(val_data), f1_score(val_labels, val_preds, average='binary', pos_label=0, zero_division=0)


    def predict_data(self, test_data, device):
        self.eval()

        test_preds = []

        with torch.no_grad():
            for batch in test_data:
                input_ids, attn_masks, _ = [b.to(device) for b in batch]

                with autocast():
                    logits = self(input_ids, attn_masks)
                    pred = 1 if torch.sigmoid(logits.squeeze(-1)).cpu().numpy()[0] >= 0.5 else 0
                    test_preds.append(pred)

        return test_preds


######  Funzioni  ######

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


######  Costanti  ######

SEED = 71936

# Dataset

DOPPIA = 0
CONCETTI = 1
Q_DOPPIA = 2
Q_CONCETTI = 3
DICT_TASK = {DOPPIA: "DOPPIA", CONCETTI: "CONCETTI", Q_DOPPIA: "Q_DOPPIA", Q_CONCETTI: "Q_CONCETTI"}

DATASET = "dataset/dataset_train.json"
RW_EVAL_DATASET = "dataset/dataset_rw.json"
INDEX_FILE = "results/index.json"

# Model

BERT_UNCASED = 'dbmdz/bert-base-italian-uncased'
BERT_CASED = 'dbmdz/bert-base-italian-cased'
BERT_XXL_UNCASED = 'dbmdz/bert-base-italian-xxl-uncased'
BERT_XXL_CASED = 'dbmdz/bert-base-italian-xxl-cased'

MULTI_CASED = 'bert-base-multilingual-cased'
MULTI_UNCASED = 'bert-base-multilingual-uncased'
ROBERTA = 'xlm-roberta-base'
ELECTRA = 'dbmdz/electra-base-italian-xxl-cased-discriminator'
MLM = '/nfs_home/narici/mimBERTv2/pretrain/mlm-bert-italian/checkpoint-171000/'
WMLM = '/nfs_home/narici/mimBERTv2/pretrain/mlm-bert-italian-workplace-security/checkpoint-17000'


DICT_LLM = {BERT_UNCASED: "UNCASED", BERT_CASED: "CASED", BERT_XXL_UNCASED: "XXL_UNCASED", BERT_XXL_CASED: "XXL_CASED", MULTI_CASED: "MULTI_CASED", MULTI_UNCASED: "MULTI_UNCASED", ROBERTA: "ROBERTA", ELECTRA: "ELECTRA", MLM: "MLM", WMLM: "WMLM"}


# Classifier type

FF = 0
DICT_CLS = {FF: "FF"}

# Embedding type

POOLER = 0
CLS = 1
EMB_MEAN = 2
EMB_MAX = 3
EMB_MIN = 4
DICT_EMB = {POOLER: "POOLER", CLS: "CLS", EMB_MEAN: "EMB_MEAN", EMB_MAX: "EMB_MAX", EMB_MIN: "EMB_MIN"}

THRESHOLD = 80
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Grid search space.

    df_types = [DOPPIA, Q_DOPPIA, CONCETTI, Q_CONCETTI]
    llms = [MULTI_UNCASED, ROBERTA, ELECTRA, WMLM, MLM, BERT_CASED, BERT_UNCASED, BERT_XXL_CASED, BERT_XXL_UNCASED, MULTI_CASED] 
    emb_types = [EMB_MAX, EMB_MEAN, EMB_MIN, POOLER, CLS]
    max_lens = [128, 512]
    balanceds = [True, False]
    dropouts = [0.1, 0.2, 0.4]
    freezes = [True, False]


    for df_type, llm, max_len, balanced, dropout, emb_type, freeze in tqdm(itertools.product(df_types, llms, max_lens, balanceds, dropouts, emb_types, freezes), total=len(list(itertools.product(df_types, llms, max_lens, balanceds, dropouts, emb_types, freezes)))):

        set_seed(SEED)

        # Check test already done.

        with open(INDEX_FILE, 'r') as f:
            index_dict = json.load(f)

        done = False
        for k, v in index_dict.items():

            if v['task'] == DICT_TASK[df_type] and v['llm'] == DICT_LLM[llm] and v['emb_type'] == DICT_EMB[emb_type] and v['max_len'] == max_len and v['threshold'] == THRESHOLD and v['batch_size'] == BATCH_SIZE and v['lr'] == LEARNING_RATE and v['weight_decay'] == WEIGHT_DECAY and v['dropout'] == dropout and v['balanced'] == balanced and v['freeze'] == freeze:
                done = True
                break

        if done:
            continue
        

        # Not compatible combinations.
        if llm == ELECTRA and emb_type == POOLER:
            continue

        # Load & Split dataset (80% train, 20% eval, 20% test)

        df = pd.read_json(DATASET)

        ans_ids = df.iAnswerId.unique().tolist()
        random.shuffle(ans_ids)

        train_ids = ans_ids[:int(len(ans_ids)*0.8)]
        val_ids = ans_ids[int(len(ans_ids)*0.8):int(len(ans_ids)*0.9)]
        test_ids = ans_ids[int(len(ans_ids)*0.9):]

        df_train = df[df.iAnswerId.isin(train_ids)].copy()
        df_val = df[df.iAnswerId.isin(val_ids)].copy()
        df_test = df[df.iAnswerId.isin(test_ids)].copy()

        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        dataset_train = SentenceDataset(df_train, df_type, max_len, llm, balanced)
        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)

        dataset_val = SentenceDataset(df_val, df_type, max_len, llm, False)
        dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)

        dataset_test = SentenceDataset(df_test, df_type, max_len, llm, False)
        dataloader_test = DataLoader(dataset_test, batch_size=1)

        # RW Dataset

        df_rw_eval = pd.read_json(RW_EVAL_DATASET)
        dataset_rw_eval = SentenceDataset(df_rw_eval, df_type, max_len, llm, False)
        dataloader_rw_eval = DataLoader(dataset_rw_eval, batch_size=1)


        timestamp = str(round(time.time() * 1000))

        # Set-up model.

        cls_params = {'llm': llm, 'emb_type': emb_type, 'dropout_rate': dropout, 'freeze': freeze}

        model = Classifier(FF, cls_params)
        model.to(device)

        # Training.

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()

        total_steps = len(dataloader_train) * EPOCHS
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

        train_losses, f1_scores, val_losses, best_ep = model.train_cls(dataloader_train, dataloader_val, EPOCHS, criterion, optimizer, scaler, lr_scheduler, device, timestamp)

        # Testing.

        tuned = Classifier(FF, cls_params)
        tuned.load_state_dict(torch.load('results/models/' + timestamp + '.pth'))
        tuned.to(device)
        
        test_preds = tuned.predict_data(dataloader_test, device)
        test_labels = dataset_test.get_labels()

        rw_preds = tuned.predict_data(dataloader_rw_eval, device)
        rw_labels = dataset_rw_eval.get_labels()


        # Pickle save.

        if df_type in [CONCETTI, Q_CONCETTI]:
            out_dict = {'test_preds': test_preds, 'test_labels': test_labels, 'rw_preds': rw_preds, 'rw_labels': rw_labels, 'test_iAnswerId': dataset_test.get_answerids(), 'test_iConcPeso': dataset_test.get_concweigths(), 'rw_iAnswerId': dataset_rw_eval.get_answerids(), 'rw_iConcPeso': dataset_rw_eval.get_concweigths(), 'train_losses': train_losses, 'f1_scores': f1_scores, 'val_losses': val_losses}
        else:
            out_dict = {'test_preds': test_preds, 'test_labels': test_labels, 'rw_preds': rw_preds, 'rw_labels': rw_labels, 'train_losses': train_losses, 'f1_scores': f1_scores, 'val_losses': val_losses}


        with open(INDEX_FILE, 'r') as f:
            index_dict = json.load(f)

        index_dict[timestamp] = {'task': DICT_TASK[df_type], 'llm': DICT_LLM[llm], 'emb_type': DICT_EMB[emb_type], 'threshold': THRESHOLD, 'epochs': EPOCHS, 'max_len': max_len, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'dropout': dropout, 'balanced': int(balanced), 'freeze': int(freeze)}

        with open(INDEX_FILE, 'w') as f:
            json.dump(index_dict, f)

        with open('results/experiments/' + timestamp + '.pkl', 'wb') as f:
            pickle.dump(out_dict, f)

        raise Exception("Done")
