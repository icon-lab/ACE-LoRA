import torch
import pandas as pd 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_templates import *
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from loralib.utils import load_model, apply_lora
from tqdm import tqdm
from loss import *
import ast
from open_clip_patch import patch_encode_text
from timm_vit_return_attn_patch import patch_timm_vit_return_attn_scores
from bert_modeling_bert_self_attn_patch import patch_bert_self_attn

cache_dir = './.cache/hub'
os.environ['HF_HOME'] = cache_dir

def apply_ace_hgat(loss_fn, features, attn_weights, encoder="img"):
    if encoder =="img":
        edge_adapter = loss_fn.img_edge_adapter
        node_adapter = loss_fn.img_node_adapter
    elif encoder == 'text':
        edge_adapter = loss_fn.text_edge_adapter
        node_adapter = loss_fn.text_node_adapter
    else:
        raise ValueError(f"encoder must be img or text but given {encoder}")
    B, N, D = features.shape
    patches_norm = F.normalize(features[:, 1:, :], p=2, dim=-1)
    # Similarity Matrix: (B, P, P)
    sim = torch.zeros(size=(B, N, N), device=features.device)
    patch_sim = torch.bmm(patches_norm, patches_norm.transpose(1, 2)) # [B, P, P]
    sim[:, 1:, 1:] = patch_sim
    sim[:, 0, 1:] = attn_weights
    mask_logic = torch.eye(N, device=features.device).bool().unsqueeze(0).repeat(B, 1, 1)
    mask_logic[:, 1:, 0] = True
    sim = sim.masked_fill(mask_logic, -float('inf'))
    topk_vals, topk_indices = torch.topk(sim, k=5, dim=-1)
    mask_sparse = torch.full_like(sim, -float('inf'))
    mask_sparse.scatter_(-1, topk_indices, topk_vals)
    A = F.softmax(mask_sparse, dim=-1)
    A = A.masked_fill(torch.eye(N, device=features.device).bool().unsqueeze(0).repeat(B, 1, 1), 1)
    A[:, 1:, 0] = A[:, 0, 1:]

    H_edges_raw = torch.matmul(A, features)
    H_edges_refined = edge_adapter(H_edges_raw)
    H_context_raw = torch.matmul(A.transpose(1, 2), H_edges_refined)
    H_context_processed = node_adapter(H_context_raw)
    x_out = H_context_processed
    
    return x_out

def prompt_ensemble_with_ace_hgat(model, query_words, device, tokenizer, context_length, templates, loss_fn):
    query_features = []
    with torch.no_grad():
        for qw in query_words:
            query = tokenizer([temp(qw) for temp in templates], context_length=context_length).to(device)
            feature, attn_scores = model.encode_text(query, normalize=True, output_attentions=True, output_tokens=True)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)

            attn_scores = attn_scores[-1].mean(dim=1)
            attn_scores = attn_scores.mean(dim=0, keepdim=True)
            attn_weights = attn_scores[:, 0, 1:] # relationship between global token and other token embeddings
            feature = apply_ace_hgat(loss_fn, feature.unsqueeze(0), attn_weights, encoder="text")
            feature /= feature.norm()
            query_features.append(feature)
            
        return torch.cat(query_features, dim=0)

def parse_args_from_txt(args, path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = map(str.strip, line.split(':', 1))
            try:
                parsed_value = ast.literal_eval(value)
            except:
                parsed_value = value
            setattr(args, key, parsed_value)
    return args

class Args:
    pass

args = Args()

classes = ['No Finding', 'pneumothorax']

load_path = './ACE-LoRA/checkpoint/ACE_LoRA.pt'
hyperparameter_path = os.path.join(os.path.dirname(load_path), 'model_hyperparameters.txt')
args = parse_args_from_txt(args, hyperparameter_path)
args.load_path = load_path
device = torch.device('cuda:0')
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
model, preprocess = create_model_from_pretrained(model_name, cache_dir=cache_dir) 

patch_encode_text()
patch_timm_vit_return_attn_scores()
patch_bert_self_attn()
state_dict = model.state_dict()
logit_scale = state_dict['logit_scale'].exp()
model.eval()
tokenizer = get_tokenizer(model_name)
list_lora_layers = apply_lora(args, model)
model = model.to(device)

loss_fn = CLIPLossACE_HGAT(args, logit_scale, 512).to(device=device)

load_model(args, list_lora_layers, device, loss_fn)
model.eval()

auc_metric = BinaryAUROC(thresholds=None)
acc_metric = BinaryAccuracy().to(device)

context_length = 256
acc = 0
data_dir = './SIIM/full_dataset'
TEST_CSV_PATH = './SIIM/test_labels.csv'

df = pd.read_csv(TEST_CSV_PATH)
test_paths = df['ImageId'].tolist()
pred_logits_list = []
label_list = []

text_features = prompt_ensemble_with_ace_hgat(model, classes, device, tokenizer, context_length, prompt_templates, loss_fn)
text_features = F.normalize(text_features, dim=-1)
model.visual.trunk.global_pool = ''

logits_list = []

for index in tqdm(range(len(df))):
    img_path = os.path.join(data_dir, str(test_paths[index]) + '.png')
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    label = torch.zeros(len(classes), dtype=torch.int8, device=device)
    label[df['Label'][index]] = 1
    pred = torch.zeros(len(classes), dtype=torch.int8, device=device)

    with torch.no_grad():
        image_features, attn_scores = model.visual.trunk.get_attn_scores(image)
        image_features = F.normalize(model.visual.head(image_features), dim=-1)
        attn_scores = attn_scores.mean(dim=1)
        attn_weights = attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings
        image_features = apply_ace_hgat(loss_fn, image_features, attn_weights, encoder="img")
        image_features = F.normalize(image_features, dim=-1)
        logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1)
        logits_list.append(logits)
        label_list.append(label.argmax())

logits_all = torch.cat(logits_list, dim=0)   # (N, C)
labels_all = torch.stack(label_list) 
auc = auc_metric(logits_all[:, 1], labels_all)
acc = acc_metric(logits_all[:, 1], labels_all) 

print("ACC: ", acc)
print("AUC: ", auc)
