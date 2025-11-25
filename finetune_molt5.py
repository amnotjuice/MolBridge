from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import csv
import pickle
import torch
import argparse
import os
import numpy as np
import random, transformers
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
from rdkit import Chem


##############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument('--accum_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)

parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument('--save_path', type=str, default='./outputs_truepair_mb_rebuttal')
parser.add_argument('--checkpoint', type=str, default='laituan245/molt5-base')
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--direction', type=str, default='smiles2caption', choices=['smiles2caption', 'caption2smiles'])
parser.add_argument('--dataset', type=str, default='ChEBI-20_data', choices=["pubchemstm", "ChEBI-20_data"])

parser.add_argument('--eval_step', type=int, default=1000)
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()

accum_steps = args.accum_steps
batch_size = args.batch_size
learning_rate = args.learning_rate
save_path = args.save_path + '/' + args.direction + '_' + args.checkpoint.split('-')[-1]+ '/' 

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
    return smiles 

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
                if args.direction == 'smiles2caption':
                    self.descriptions[line['cid']] = line['desc']
                    self.smiles[line['cid']] = 'Provide a whole description of this molecule: ' + canonicalize_smiles(line['smiles']) # typo
                    self.cids.append(line['cid'])
                    count += 1
                else:
                    self.descriptions[line['cid']] = canonicalize_smiles(line['smiles'])
                    self.smiles[line['cid']] = 'Provide a molecule based on this description: ' + line['desc']
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
                'description':description,
                'smiles':smiles
                }     


def collate_fn(batch):
    description, smiles = [], []
    for b in batch:
        description.append(b['description'])
        smiles.append(b['smiles'])
        
    return description, smiles

train_data = ChEBI_20_data_Dataset(
    args.data_dir,
    args.dataset,
    'train',
)

valid_data = ChEBI_20_data_Dataset(
    args.data_dir,
    args.dataset,
    'validation',
)

test_data = ChEBI_20_data_Dataset(
    args.data_dir,
    args.dataset,
    'test',
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
if not valid_data == None:
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
else:
    valid_loader=None
    
if not test_data == None:
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
else:
    test_loader=None

print('DONE!!!')
##############################################################################################################
print('PREPARING MODEL...')
tokenizer = T5Tokenizer.from_pretrained('laituan245/molt5-base', model_max_length=256) # 512
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to('cuda')

model.to(device)
print('DONE!!!')
##############################################################################################################
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

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

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
if args.eval_only:
    epoch = 0
    validate(model, valid_loader, 0)

total_finetuning_steps = 50000
epoch = math.ceil(total_finetuning_steps / (len(train_loader) / accum_steps)) + 1
epoch = 50

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
        description, smiles = batch

        description_token = tokenizer(description, return_tensors='pt', padding='longest', truncation=True).to(model.device)
        smiles_token = tokenizer(smiles, return_tensors='pt', padding='longest', truncation=True).to(model.device)

        labels = description_token['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                input_ids=smiles_token['input_ids'],
                attention_mask=smiles_token['attention_mask'],
                labels=labels,
                use_cache=False 
                )

            loss = output.loss / accum_steps
        # loss.backward()
        accum_loss += loss.item()
        scaler.scale(loss).backward()

        # if ((i + 1) % accum_steps == 0) or (i + 1 == len(train_loader)):
        if ((i + 1) % accum_steps == 0):
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
    
    if (ep+1) % 10 == 0:
        model.save_pretrained(output_dir+f'/{ep+1}_model')

    writer.add_scalar("Loss/train_epoch", total_loss/current_steps, ep+1)
    score = validate(model, valid_loader, ep+1)
    if score < best_score:
        print(f'Best model on epoch {ep+1}')
        # torch.save(model.state_dict(), output_dir+f'/best_model')
        model.save_pretrained(output_dir+f'/best_model')
        best_score = score
    writer.flush()


if not args.eval_only:
    writer.close()
