# MolBridge

**Bridging the Gap Between Molecule and Textual Descriptions via Substructure-aware Alignment**  
*EMNLP 2025 Main Conference*

MolBridge is a substructure-aware generative model designed to align molecular structures and textual descriptions.  
We release two complementary checkpoints for bidirectional generation between molecule representations and natural language captions.

---

## 🔗 Released Models

| Model | Description | Link |
|:------|:-------------|:------|
| **MolBridge-Gen-Base-C2S** | Caption → SMILES generation | [PhTae/MolBridge-Gen-Base-C2S](https://huggingface.co/PhTae/MolBridge-Gen-Base-C2S) |
| **MolBridge-Gen-Base-S2C** | SMILES → Caption generation | [PhTae/MolBridge-Gen-Base-S2C](https://huggingface.co/PhTae/MolBridge-Gen-Base-S2C) |

---

## 🧪 Usage Examples

### Caption → SMILES (C2S)
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("PhTae/MolBridge-Gen-Base-C2S")

caption = "The molecule is a monoterpene that is bicyclo[2.2.1]heptane substituted by methyl groups at positions 1, 3 and 3. It is a monoterpene, a terpenoid fundamental parent and a carbobicyclic compound."
caption = "Provide a molecule based on this description: " + caption

token = tokenizer(caption, return_tensors="pt", padding="longest", truncation=True)
gen_results = model.generate(
    input_ids=token["input_ids"],
    attention_mask=token["attention_mask"],
    num_beams=5,
    max_new_tokens=512
)
```

### SMILES → Caption (S2C)
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("PhTae/MolBridge-Gen-Base-S2C")

canonicalized_smiles = "CC(=O)N[C@@H](CCCN=C(N)N)C(=O)[O-]"
canonicalized_smiles = "Provide a whole description of this molecule: " + canonicalized_smiles

token = tokenizer(canonicalized_smiles, return_tensors="pt", padding="longest", truncation=True)
gen_results = model.generate(
    input_ids=token["input_ids"],
    attention_mask=token["attention_mask"],
    num_beams=5,
    max_new_tokens=512
)

```
