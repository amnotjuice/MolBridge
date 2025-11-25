from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import csv
import pickle
import torch
import argparse
import json
import numpy as np
import random
from tqdm import tqdm
from rdkit import Chem


##############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--num_workers", type=int, default=8)

# parser.add_argument('--save_path', type=str, default='./output_mine/')
parser.add_argument('--checkpoint', type=str, default="./outputs_truepair/smiles2caption_base/2025-04-25-15-03-26_8_16/best_model")
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--direction', type=str, default='smiles2caption', choices=['smiles2caption', 'caption2smiles'])
parser.add_argument('--dataset', type=str, default='ChEBI-20_data', choices=["pubchemstm", "ChEBI-20_data"])


args = parser.parse_args()

batch_size = args.batch_size
save_path = args.checkpoint

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
        ):
        self.data_path = data_path
        self.cids = []
        self.descriptions = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                if args.direction == 'smiles2caption':
                    self.descriptions[line['cid']] = line['desc']
                    self.smiles[line['cid']] = 'Provide a whole description of this molecule: ' + canonicalize_smiles(line['smiles'])
                    self.cids.append(line['cid'])
                else:
                    self.descriptions[line['cid']] = canonicalize_smiles(line['smiles'])
                    self.smiles[line['cid']] = 'Provide a molecule based on this description: ' + line['desc']
                    self.cids.append(line['cid'])

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


class PubChem_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path

        #load data
        with open (osp.join(data_path, dataset, split+'.pkl'),"rb") as f:
            self.data = pickle.load(f)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]["smiles"]
        description = self.data[idx]["text"]

        return {
                'description':description,
                'smiles':smiles
                }
    
def collate_fn(batch):
    description, smiles = [], []
    for b in batch:
        description.append(b['description'])
        smiles.append(b['smiles'])
        
    return description, smiles


test_data = ChEBI_20_data_Dataset(
    args.data_dir,
    args.dataset,
    'test',
)


test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn
)

print('DONE!!!')
##############################################################################################################
print('PREPARING MODEL...')
tokenizer = T5Tokenizer.from_pretrained('laituan245/molt5-base', model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to('cuda')
model.eval()
print('DONE!!!')
##############################################################################################################
@torch.no_grad()
def inference(model, dataloader):
    results = []
    gt = []
    for i, batch in enumerate(tqdm(dataloader)):
        description, smiles = batch

        gt += description

        smiles_token = tokenizer(smiles, return_tensors='pt', padding='longest', truncation=True).to(model.device)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        gen_results = model.generate(input_ids=smiles_token['input_ids'],
                                    attention_mask=smiles_token['attention_mask'], 
                                    num_beams=5, 
                                    max_new_tokens=512)
    
        for jjjjj, gen in enumerate(gen_results):
            gen = tokenizer.decode(gen, skip_special_tokens=True)
            results.append({smiles[jjjjj]: gen})
        
        if i < 10:
            print(gen)
    return results, gt

##############################################################################################################
results, gt = inference(model, test_loader)

import os
with open(os.path.join(save_path, f'{args.direction}.json'), 'w') as f:
    json.dump(results, f, indent=2)
