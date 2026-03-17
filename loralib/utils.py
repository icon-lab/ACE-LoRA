import os

import torch
import torch.nn as nn
from typing import Dict

from .layers import LoRALayer, AttentionLoRA, BertAttentionLoRA
from timm.models.vision_transformer import Attention
from transformers.models.bert.modeling_bert import BertAttention


INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'top': [11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    indices = INDEX_POSITIONS_TEXT[args.position]
    text_encoder = clip_model.text.transformer.encoder
    for i, block in enumerate(text_encoder.layer):
        if i in indices:
            for name, submodule in block.named_children():
                if isinstance(submodule, BertAttention):
                    new_multi_head_lora = BertAttentionLoRA(
                        submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate, seed=args.seed)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)

    indices = INDEX_POSITIONS_VISION[args.position]
    vision_encoder = clip_model.visual.trunk
    for i, block in enumerate(vision_encoder.blocks):
        if i in indices:
            for name, submodule in block.named_children():
                if isinstance(submodule, Attention):
                    new_multi_head_lora = AttentionLoRA(
                        submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate, seed=args.seed)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def save_lora(args, list_lora_layers, loss_fn, msg, save_dir):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights
    
    if args.loss_type == 'clip_loss_ace_hgnn':
        weights['img_edge_adapter'] = loss_fn.img_edge_adapter.state_dict()
        weights['img_node_adapter'] = loss_fn.img_node_adapter.state_dict()
        weights['text_edge_adapter'] = loss_fn.text_edge_adapter.state_dict()
        weights['text_node_adapter'] = loss_fn.text_node_adapter.state_dict()
    
    if args.learnable_logit_scale:
        weights['logit_scale'] = loss_fn.logit_scale.data.cpu()

    metadata = {
        'r': args.r,
        'topk': args.topk,
        'params': args.params,
        'position': args.position,
        'loss_type' : args.loss_type,
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    save_path = f'{save_dir}/{args.filename}_{msg}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def load_model(args, list_lora_layers, device, loss_fn=None):

    if not os.path.exists(args.load_path):
        raise FileNotFoundError(f'File {args.load_path} does not exist.')

    loaded_data = torch.load(args.load_path, map_location=device)

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    if args.loss_type == 'clip_loss_ace_hgnn':
        loss_fn.img_edge_adapter.load_state_dict(weights['img_edge_adapter'])
        loss_fn.img_node_adapter.load_state_dict(weights['img_node_adapter'])
        loss_fn.text_edge_adapter.load_state_dict(weights['text_edge_adapter'])
        loss_fn.text_node_adapter.load_state_dict(weights['text_node_adapter'])

    if args.learnable_logit_scale:
        loss_fn.logit_scale.data.copy_(weights['logit_scale'])

    print(f'LoRA weights loaded from {args.load_path}')
