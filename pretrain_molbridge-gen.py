from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import csv
import pickle
import torch
import argparse
import os
import numpy as np
import json
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem


##############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument('--accum_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=5e-4)

parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument('--save_path', type=str, default='./outputs_pretrained_truepair')
parser.add_argument('--checkpoint', type=str, default='laituan245/molt5-base')
parser.add_argument('--data_dir', type=str, default=YOUR_DATA_DIR_PATH)
parser.add_argument('--dataset', type=str, default='ChEBI-20_data', choices=["pubchemstm", "ChEBI-20_data"])

parser.add_argument('--eval_step', type=int, default=1000)
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()

accum_steps = args.accum_steps
batch_size = args.batch_size
learning_rate = args.learning_rate
save_path = args.save_path

##############################################################################################################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 42
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

##############################################################################################################
import time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
output_dir = os.path.join(save_path, f'{current_time}_{args.batch_size}_{args.accum_steps}')
if args.eval_only:
    writer = None
else:
    writer = SummaryWriter(output_dir)

##############################################################################################################
print('PREPARING DATA...')

def canonicalize_smiles(smiles, canonical=True, isomericSmiles=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=canonical)#, isomericSmiles=isomericSmiles)
    return smiles  # 유효하지 않은 SMILES 처리

class ChEBI_20_data_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        add_subkey=False
        ):
        self.data_path = data_path
        self.cids = []
        self.descriptions = {}
        self.smiles = {}
        
        count = 0
        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                # if args.direction == 'smiles2caption':
                #     self.descriptions[line['cid']] = line['desc']
                #     self.smiles[line['cid']] = 'Provide a whole descriptions of this molecule: ' + line['smiles']
                #     self.cids.append(line['cid'])
                #     count += 1
                # else:
                #     self.descriptions[line['cid']] = line['smiles']
                #     self.smiles[line['cid']] = 'Provide a molecule based on this description: ' + line['desc']
                #     self.cids.append(line['cid'])
                #     count += 1
                
                self.descriptions[line['cid']] = line['desc']
                self.smiles[line['cid']] = 'Provide a whole descriptions of this molecule: ' + canonicalize_smiles(line['smiles'])
                self.cids.append(line['cid'])
                count += 1

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        cid = self.cids[idx]
        smiles = self.smiles[cid]
        description = self.descriptions[cid]
        
        return {
                'cid':cid,
                'outputs':description,
                'inputs':smiles
                }  

class My_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        add_subkey=False,
        ):
        self.data_path = data_path
        self.cids = []
        self.output = {}
        self.input = {}

        count = 0
        #load data

        already = {}

        data = json.load(open('/data/user16/MolT5/subgraph-keyword_truepairs.json'))
        for n, line in enumerate(data):
                self.output[count] = line['caption']
                self.input[count] = 'Provide a keyword of this substructure: ' + canonicalize_smiles(line['smiles'])
                self.cids.append(count)
                count += 1
  
                self.output[count] = canonicalize_smiles(line['smiles'])
                self.input[count] = 'Provide a substructure based on this keyword: ' + line['caption']
                
                self.cids.append(count)
                count += 1

                if line['caption'] + line['smiles'] not in already.keys():
                    self.output[count] = line['caption']
                    self.input[count] = 'Provide a whole description of this molecule: ' + canonicalize_smiles(line['smiles'])
                    self.cids.append(count)
                    count += 1
    
                    self.output[count] = canonicalize_smiles(line['smiles'])
                    self.input[count] = 'Provide a molecule based on this description: ' + line['caption']
                    
                    self.cids.append(count)
                    count += 1
                    
                    already[line['caption'] + line['smiles']] = 1
                else:
                    continue

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        cid = self.cids[idx]
        inputs = self.input[cid]
        outputs = self.output[cid]
        
        return {
                'cid':cid,
                'inputs':inputs,
                'outputs':outputs
                }        

    
def collate_fn(batch):
    outputs, inputs = [], []
    for b in batch:
        inputs.append(b['inputs'])
        outputs.append(b['outputs'])
        
    return outputs, inputs

train_data = My_Dataset(
    args.data_dir,
    args.dataset,
    'train',
    add_subkey=True,
)

valid_data = ChEBI_20_data_Dataset(
    args.data_dir,
    args.dataset,
    'validation',
)


train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True,
    shuffle=True,
    drop_last=False,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

print('DONE!!!')
##############################################################################################################
print('PREPARING MODEL...')
# max_len = 512 if 'base' in args.checkpoint else 256
# max_len = 256
max_len = 128
tokenizer = T5Tokenizer.from_pretrained('laituan245/molt5-base', model_max_length=max_len)
# model = T5ForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to('cuda')
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to('cuda')

model.to(device)
print('DONE!!!')
##############################################################################################################
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = transformers.Adafactor(model.parameters(), lr=learning_rate, relative_step=False, warmup_init=False)
optimizer.zero_grad()
# scaler = torch.cuda.amp.GradScaler()
scaler = torch.amp.GradScaler()
##############################################################################################################
@torch.no_grad()
def validate(model, dataloader, epoch):
    total_loss = 0
    current_steps = 0

    for i, batch in enumerate(tqdm(dataloader)):
        description, smiles = batch

        description_token = tokenizer(description, return_tensors='pt', padding='longest', truncation=True).to(model.device)
        smiles_token = tokenizer(smiles, return_tensors='pt', padding='longest', truncation=True).to(model.device)

        labels = description_token['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        # print(description, smiles)
        output = model(
            input_ids=smiles_token['input_ids'],
            attention_mask=smiles_token['attention_mask'],
            labels=labels,
            use_cache=False 
            )

        total_loss += output.loss.item()
        current_steps += 1


    prefix = 'Loss/val_epoch'
    
    if not args.eval_only:
        writer.add_scalar(prefix, total_loss/current_steps, epoch)
        writer.flush()
    print(f'Loss: {total_loss / current_steps}')
    
    return total_loss / current_steps

##############################################################################################################

epoch = 50
a = 0
print('Start train the model for {} epochs...'.format(epoch))
accum_loss = 0
current_steps = 0
best_score = 99999
for ep in range(epoch):
    total_loss = 0
    total_steps = 0
    print(f'Start {ep+1} th epoch...')
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        outputs, inputs = batch

        inputs_token = tokenizer(inputs, return_tensors='pt', padding='longest', truncation=True).to(model.device)
        outputs_token = tokenizer(outputs, return_tensors='pt', padding='longest', truncation=True).to(model.device)

        labels = outputs_token['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                input_ids=inputs_token['input_ids'],
                attention_mask=inputs_token['attention_mask'],
                labels=labels,
                use_cache=False 
                )

            loss = output.loss / accum_steps
        accum_loss += loss.item()
        scaler.scale(loss).backward()
        # loss.backward()
        
        a += 1

        # if ((i + 1) % accum_steps == 0) or (i + 1 == len(train_loader)):
        if a % accum_steps == 0:
            total_loss += accum_loss
            total_steps += 1
            current_steps += 1

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/train", accum_loss, current_steps)
            writer.flush()

            accum_loss = 0
            
            # break
            
    writer.add_scalar("Loss/train_epoch", total_loss/current_steps, ep+1)
    score = validate(model, valid_loader, ep+1)
    if score < best_score:
        print(f'Best model on epoch {ep+1}')
        model.save_pretrained(output_dir+f'/best_model')
        best_score = score

    writer.flush()

    if (ep+1) % 3 == 0:
        model.save_pretrained(output_dir+f'/{ep+1}_model')


if not args.eval_only:
    writer.close()
