import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
import torch.distributed as dist
import os
from torchmetrics.classification import MulticlassAUROC, BinaryAUROC, MultilabelAUROC
import pandas as pd
from PIL import Image
from prompt_templates import *
import numpy as np
import pydicom
import ast
import torch.nn.functional as F
import datetime

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '48933'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=2700))
    torch.cuda.set_device(rank) 

def prompt_ensemble(model, query_words, device, tokenizer, context_length, templates):
    query_features = []
    with torch.no_grad():
        for qw in query_words:
            query = tokenizer([temp(qw) for temp in templates], context_length=context_length).to(device)
            feature = model.encode_text(query, normalize=True)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            query_features.append(feature.unsqueeze(0))
    
        return torch.cat(query_features, dim=0)
    
def prompt_ensemble_with_ace_hgat(model, query_words, device, tokenizer, context_length, templates, args, loss_fn):
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

            feature = apply_ace_hgat(args, loss_fn, feature.unsqueeze(0), attn_weights, encoder="text")
            feature /= feature.norm()
            query_features.append(feature)
            
        return torch.cat(query_features, dim=0)

def apply_ace_hgat(args, loss_fn, features, attn_weights, encoder="img"):
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
    topk_vals, topk_indices = torch.topk(sim, k=args.topk, dim=-1)
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

def zero_shot_chexpert_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu):
    
    all_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']
    
    classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    map_indices = [all_classes.index(item) for item in classes]
    if trained_on_multi_gpu:
        device = args.rank
        clip_model = clip_model.module
    else:
        device = args.device
    clip_model.eval()

    auc_metric = MulticlassAUROC(num_classes=len(classes), average="macro", thresholds=None)

    context_length = args.context_length
    acc = 0
    data_dir = './chexpertchestxrays-u20210408'
    TEST_CSV_PATH = './chexpert_5x200.csv'

    df = pd.read_csv(TEST_CSV_PATH)
    test_paths = df['Path'].tolist()
    pred_logits_list = []
    label_list = []
    
    if args.loss_type == 'clip_loss_ace_hgnn' and (args.apply_gnn_encoders == 'text' or args.apply_gnn_encoders == 'both'):
        text_features = prompt_ensemble_with_ace_hgat(clip_model, classes, device, tokenizer, context_length, prompt_templates, args, loss_fn)
        text_features = F.normalize(text_features, dim=-1)

    elif args.loss_type == 'clip_loss' or args.apply_gnn_encoders == 'vision':
        text_features = prompt_ensemble(clip_model, classes, device, tokenizer, context_length, prompt_templates)
        text_features = F.normalize(text_features, dim=-1)

    for index in tqdm(range(len(df))):
        img_path = os.path.join(data_dir, test_paths[index])
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        label = torch.from_numpy(df.iloc[index, 6:-1].values.astype(np.int8)).to(device)
        label = label[map_indices]
        pred = torch.zeros(len(classes), dtype=torch.int8, device=device)

        with torch.no_grad():
            if args.loss_type == 'clip_loss_ace_hgnn':
                clip_model.visual.trunk.global_pool = ''
                image_features, attn_scores = clip_model.visual.trunk.get_attn_scores(image)
                image_features = F.normalize(clip_model.visual.head(image_features), dim=-1)
                attn_scores = attn_scores.mean(dim=1)
                attn_weights = attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings  
                
                if args.apply_gnn_encoders == 'vision':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features.t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'text':
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'both':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

            else:
                image_features = clip_model.encode_image(image, normalize=True)
                logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1).squeeze()
            
            predicted_class = logits.argmax()
            pred[predicted_class] = 1
            label_list.append(label.argmax())
            pred_logits_list.append(logits)

            if torch.all(pred == label):
                acc += 1

    acc /= len(df)
    auc = auc_metric(torch.stack(pred_logits_list), torch.stack(label_list))
    print('Chexpert 5x200 Accuracy : ', acc)
    print('Chexpert 5x200 Area Under Curve : ', auc)

    return acc, auc

def zero_shot_rsna_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu):
    
    classes = ['No Finding', 'pneumonia']
    
    if trained_on_multi_gpu:
        device = args.rank
        clip_model = clip_model.module
    else:
        device = args.device
    clip_model.eval()

    auc_metric = BinaryAUROC(thresholds=None)
    context_length = args.context_length
    acc = 0
    TEST_CSV_PATH = './RSNA/test.csv'

    df = pd.read_csv(TEST_CSV_PATH)
    test_paths = df['Path'].tolist()
    pred_logits_list = []
    label_list = []

    if args.loss_type == 'clip_loss_ace_hgnn' and (args.apply_gnn_encoders == 'text' or args.apply_gnn_encoders == 'both'):
        text_features = prompt_ensemble_with_ace_hgat(clip_model, classes, device, tokenizer, context_length, prompt_templates, args, loss_fn)
        text_features = F.normalize(text_features, dim=-1)

    elif args.loss_type == 'clip_loss' or args.apply_gnn_encoders == 'vision':
        text_features = prompt_ensemble(clip_model, classes, device, tokenizer, context_length, prompt_templates)
        text_features = F.normalize(text_features, dim=-1)

    for index in tqdm(range(len(df))):
        img_path = test_paths[index]
        img_data = pydicom.dcmread(img_path).pixel_array
        image = preprocess(Image.fromarray(img_data)).unsqueeze(0).to(device)

        label = torch.zeros(len(classes), dtype=torch.int8, device=device)
        label[df['Target'][index]] = 1
        pred = torch.zeros(len(classes), dtype=torch.int8, device=device)

        with torch.no_grad():
            if args.loss_type == 'clip_loss_ace_hgnn':
                clip_model.visual.trunk.global_pool = ''
                image_features, attn_scores = clip_model.visual.trunk.get_attn_scores(image)
                image_features = F.normalize(clip_model.visual.head(image_features), dim=-1)
                attn_scores = attn_scores.mean(dim=1)
                attn_weights = attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings  

                if args.apply_gnn_encoders == 'vision':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features.t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'text':
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'both':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

            else:
                image_features = clip_model.encode_image(image, normalize=True)
                logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1).squeeze()
            
            predicted_class = logits.argmax()
            pred[predicted_class] = 1
            label_list.append(label)
            pred_logits_list.append(logits)

            if torch.all(pred == label):
                acc += 1

    acc /= len(df)
    auc = auc_metric(torch.stack(pred_logits_list), torch.stack(label_list))
    print('RSNA Accuracy : ', acc)
    print('RSNA Area Under Curve : ', auc)

    return acc, auc
   
def zero_shot_chest_xray_14_eval(args, clip_model, tokenizer, loss_fn, preprocess, trained_on_multi_gpu):
        
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    if trained_on_multi_gpu:
        device = args.rank
        clip_model = clip_model.module
    else:
        device = args.device
    clip_model.eval()

    auc_metric = MultilabelAUROC(num_labels=len(classes), average="macro", thresholds=None)

    context_length = args.context_length
    data_dir = './Chest_x_ray_14/images'
    TEST_CSV_PATH = './Chest_x_ray_14/test_set_biomedclip_hot_encode.csv'

    df = pd.read_csv(TEST_CSV_PATH)
    df['Label Indices'] = df['Label Indices'].apply(ast.literal_eval)
    test_paths = df['Image Index'].tolist()
    pred_logits_list = []
    label_list = []

    if args.loss_type == 'clip_loss_ace_hgnn' and (args.apply_gnn_encoders == 'text' or args.apply_gnn_encoders == 'both'):
        text_features = prompt_ensemble_with_ace_hgat(clip_model, classes, device, tokenizer, context_length, prompt_templates, args, loss_fn)
        text_features = F.normalize(text_features, dim=-1)

    elif args.loss_type == 'clip_loss' or args.apply_gnn_encoders == 'vision':
        text_features = prompt_ensemble(clip_model, classes, device, tokenizer, context_length, prompt_templates)
        text_features = F.normalize(text_features, dim=-1)

    for index in tqdm(range(len(df))):
        img_path = os.path.join(data_dir, test_paths[index])
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        label = torch.tensor(df.loc[index, 'Label Indices'], dtype=torch.int8, device=device)

        with torch.no_grad():
            if args.loss_type == 'clip_loss_ace_hgnn':
                clip_model.visual.trunk.global_pool = ''
                image_features, attn_scores = clip_model.visual.trunk.get_attn_scores(image)
                image_features = F.normalize(clip_model.visual.head(image_features), dim=-1)
                attn_scores = attn_scores.mean(dim=1)
                attn_weights = attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings  

                if args.apply_gnn_encoders == 'vision':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)                    
                    logits = (image_features[:, 0] @ text_features.t())

                elif args.apply_gnn_encoders == 'text':
                    logits = (image_features[:, 0] @ text_features[:, 0].t())

                elif args.apply_gnn_encoders == 'both':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (image_features[:, 0] @ text_features[:, 0].t())
                
            else:
                image_features = clip_model.encode_image(image, normalize=True)
                logits = image_features @ text_features.t()
                        
            min_val = torch.exp(logits).min() 
            max_val = torch.exp(logits).max()
            logits = (torch.exp(logits) - min_val) / (max_val - min_val) 
            label_list.append(label)
            pred_logits_list.append(logits.squeeze())

    auc = auc_metric(torch.stack(pred_logits_list),torch.stack(label_list))
    print('NIH Chest X-Ray 14 Area Under Curve : ', auc)

    return auc

def zero_shot_siim_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu):
    
    classes = ['No Finding', 'pneumothorax']
    
    if trained_on_multi_gpu:
        device = args.rank
        clip_model = clip_model.module
    else:
        device = args.device
    clip_model.eval()

    auc_metric = BinaryAUROC(thresholds=None)
    context_length = args.context_length
    acc = 0
    data_dir = './SIIM/full_dataset'
    TEST_CSV_PATH = './SIIM/test_labels.csv'

    df = pd.read_csv(TEST_CSV_PATH)
    test_paths = df['ImageId'].tolist()
    pred_logits_list = []
    label_list = []

    if args.loss_type == 'clip_loss_ace_hgnn' and (args.apply_gnn_encoders == 'text' or args.apply_gnn_encoders == 'both'):
        text_features = prompt_ensemble_with_ace_hgat(clip_model, classes, device, tokenizer, context_length, prompt_templates, args, loss_fn)
        text_features = F.normalize(text_features, dim=-1)

    elif args.loss_type == 'clip_loss' or args.apply_gnn_encoders == 'vision':
        text_features = prompt_ensemble(clip_model, classes, device, tokenizer, context_length, prompt_templates)
        text_features = F.normalize(text_features, dim=-1)

    for index in tqdm(range(len(df))):
        img_path = os.path.join(data_dir, str(test_paths[index]) + '.png')
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        label = torch.zeros(len(classes), dtype=torch.int8, device=device)
        label[df['Label'][index]] = 1
        pred = torch.zeros(len(classes), dtype=torch.int8, device=device)

        with torch.no_grad():
            if args.loss_type == 'clip_loss_ace_hgnn':
                clip_model.visual.trunk.global_pool = ''
                image_features, attn_scores = clip_model.visual.trunk.get_attn_scores(image)
                image_features = F.normalize(clip_model.visual.head(image_features), dim=-1)
                attn_scores = attn_scores.mean(dim=1)
                attn_weights = attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings  

                if args.apply_gnn_encoders == 'vision':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features.t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'text':
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

                elif args.apply_gnn_encoders == 'both':
                    image_features = apply_ace_hgat(args, loss_fn, image_features, attn_weights.unsqueeze(0), encoder="img")
                    image_features = F.normalize(image_features, dim=-1)
                    logits = (logit_scale * image_features[:, 0] @ text_features[:, 0].t()).detach().softmax(dim=-1).squeeze()

            else:
                image_features = clip_model.encode_image(image, normalize=True)
                logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1).squeeze()
            
            predicted_class = logits.argmax()
            pred[predicted_class] = 1
            label_list.append(label)
            pred_logits_list.append(logits)

            if torch.all(pred == label):
                acc += 1

    acc /= len(df)
    auc = auc_metric(torch.stack(pred_logits_list), torch.stack(label_list))
    print('SIIM Accuracy : ', acc)
    print('SIIM Area Under Curve : ', auc)

    return acc, auc