#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from torch.jit import Final
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Attention
from transformers.models.bert.modeling_bert import BertAttention
from typing import Optional, Tuple

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        fan_in_fan_out: bool = False,
        dropout_rate:float = 0,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            #self.scaling = self.lora_alpha / self.r
            self.scaling = self.lora_alpha/math.sqrt(self.r) # 
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            self.register_parameter(f'{lora_name}_lora_B', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
                
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling
    
    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling
            
    
    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True 


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        nn.Embedding.reset_parameters(self)
        self.init_lora_param()

    def init_lora_param(self):
        if hasattr(self, 'w_lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.w_lora_A)
            nn.init.normal_(self.w_lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_train(mode)
        
    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Embedding.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Embedding.forward(self, x, **kwargs)

class LinearLoRA(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        seed: int = 1,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def train(self, mode: bool = True):
        super().train(mode)     
        self.lora_train(mode)

        
    def forward(self, x: torch.Tensor, **kwargs):
        
        if self.dropout is None: # do as before
            if self.r > 0 and not self.merged:
                self.merge_lora_param()
                result = nn.Linear.forward(self, x, **kwargs)
                self.sub_lora_data()
                return result
            else:
                return nn.Linear.forward(self, x, **kwargs)
            
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)

        if self.training and self.dropout.p > 0:
            x = self.dropout(x)
        
        if self.r > 0 and not self.merged:
            lora_adjustment = torch.matmul(x,self.merge_BA('weight').transpose(0, 1)) * self.scaling 
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result

class Conv1d(nn.Conv1d, LoRALayer):
    # LoRA implemented in a Conv1d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv1d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv1d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv1d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv1d.forward(self, x, **kwargs)

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)

class Conv3d(nn.Conv3d, LoRALayer):
    # LoRA implemented in a Conv3d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv3d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv3d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv3d.forward(self, x, **kwargs)


class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            r: int = 0, 
            lora_alpha: int = 1, 
            dropout_rate:float = 0.,
            **kwargs
        ):
        super().__init__()
        
        self.dropout = 0 # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        #self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=existing_mha.in_proj_bias is not None)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Initialize parameters
        with torch.no_grad():
            
            # Extract the existing weights and biases
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            # Initialize q_proj
            self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

            # Initialize k_proj
            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

            # Initialize v_proj
            self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

            # Initialize proj
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention
        
        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
        
        # Init qkv as a new lora linear layer 
        for item in enable_lora:
            if item == 'q':
                self.q_proj = LinearLoRA(self.q_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'k':
                self.k_proj = LinearLoRA(self.k_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'v':
                self.v_proj = LinearLoRA(self.v_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
            elif item == 'o':
                self.proj = LinearLoRA(self.proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate)
        
    def forward_module(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        #self.lora_train(mode)  

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):
        

        return self.forward_module(query, key, value, **kwargs) 
        
class AttentionLoRA(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            existing_mha: Attention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            r: int = 0, 
            lora_alpha: int = 1, 
            dropout_rate: float = 0.,
            seed: int = 1,
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.embed_dim = existing_mha.proj.in_features
        self.num_heads = existing_mha.num_heads
        self.head_dim = existing_mha.head_dim
        assert self.embed_dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.dropout = 0
        self.q_norm = existing_mha.q_norm
        self.k_norm = existing_mha.k_norm
        self.attn_drop = nn.Dropout(self.dropout)
        self.proj_drop = nn.Dropout(self.dropout)
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        self.enable_lora = enable_lora
        self.seed = seed
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.qkv.bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.qkv.bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.qkv.bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.proj.bias is not None)

        # Initialize parameters
        with torch.no_grad():
            existing_weight = existing_mha.qkv.weight.data
            existing_bias = existing_mha.qkv.bias.data

            self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

            self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

            self.proj.weight.data.copy_(existing_mha.proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.proj.bias.data)

        self.q_proj, self.k_proj, self.v_proj, self.proj = self.inject_lora(self.q_proj, self.k_proj, self.v_proj, self.proj)


    def inject_lora(self, q, k, v, proj):
        for item in self.enable_lora:
            if item == 'q':
                q = LinearLoRA(q,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'k':
                k = LinearLoRA(k,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'v':
                v = LinearLoRA(v,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'o':
                proj = LinearLoRA(proj,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
                
        return q, k, v, proj

    def forward(self, x: torch.Tensor, return_attn_scores=False) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_attn_scores:
            q = q * self.scale
            attn_scores = q @ k.transpose(-2, -1)
            attn = attn_scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return (x, attn_scores)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BertAttentionLoRA(nn.Module):
    def __init__(self,
                 existing_mha: BertAttention, 
                 enable_lora: list = ['q', 'k', 'v', 'o'],
                 r: int = 0, 
                 lora_alpha: int = 1, 
                 dropout_rate: float = 0.,
                 seed:int = 1,):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.self_attn = existing_mha.self
        self.output = existing_mha.output
        self.num_attention_heads = self.self_attn.num_attention_heads
        self.attention_head_size = self.self_attn.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = self.self_attn.query.in_features

        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.proj = nn.Linear(self.output.dense.in_features, self.output.dense.in_features)
        self.LayerNorm = self.output.LayerNorm
        self.dropout = nn.Dropout(0)

        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        self.enable_lora = enable_lora
        self.seed = seed
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        # Initialize parameters
        with torch.no_grad():

            self.q_proj.weight.data.copy_(self.self_attn.query.weight.data)
            if self.self_attn.query.bias.data is not None:
                self.q_proj.bias.data.copy_(self.self_attn.query.bias.data)

            self.k_proj.weight.data.copy_(self.self_attn.key.weight.data)
            if self.self_attn.key.bias.data is not None:
                self.k_proj.bias.data.copy_(self.self_attn.key.bias.data)

            self.v_proj.weight.data.copy_(self.self_attn.value.weight.data)
            if self.self_attn.value.bias.data is not None:
                self.v_proj.bias.data.copy_(self.self_attn.value.bias.data)

            self.proj.weight.data.copy_(self.output.dense.weight.data)
            if self.output.dense.bias.data is not None:
                self.proj.bias.data.copy_(self.output.dense.bias.data)

        self.q_proj, self.k_proj, self.v_proj, self.proj = self.inject_lora(self.q_proj, self.k_proj, self.v_proj, self.proj)

        self.position_embedding_type = self.self_attn.position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = self.self_attn.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * self.self_attn.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = self.self_attn.is_decoder

    def inject_lora(self, q, k, v, proj):
        for item in self.enable_lora:
            if item == 'q':
                q = LinearLoRA(q,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'k':
                k = LinearLoRA(k,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'v':
                v = LinearLoRA(v,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
            elif item == 'o':
                proj = LinearLoRA(proj,
                                r=self.r,
                                lora_alpha=self.lora_alpha,
                                fan_in_fan_out=False,
                                dropout_rate = self.dropout_rate,
                                seed=self.seed)
                
        return q, k, v, proj

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.q_proj(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.k_proj(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        self_attn_outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            self_attn_outputs = self_attn_outputs + (past_key_value,)

        # attention_output = self.output(self_outputs[0], hidden_states)
        self_outputs = self.proj(self_attn_outputs[0])
        attention_output = self.LayerNorm(self_outputs + hidden_states)
        outputs = (attention_output,) + self_attn_outputs[1:]  # add attentions if we output them
        return outputs


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0 and any(enable_lora):
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        nn.Linear.reset_parameters(self)
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        delta_w = F.conv1d(
            eval(f'self.{lora_name}_lora_A').unsqueeze(0), 
            eval(f'self.{lora_name}_lora_B').unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return self.transpose(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_train(mode)        

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Linear.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Linear.forward(self, x, **kwargs)