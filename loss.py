import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from run_utils import set_random_seed

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class CLIPLoss(nn.Module):
    def __init__(self, args, logit_scale):
        super(CLIPLoss, self).__init__()
        self.args = args
        if args.learnable_logit_scale:
            self.logit_scale = nn.Parameter(logit_scale.clone().detach())
        else:
            self.register_buffer('logit_scale', logit_scale.clone().detach())
        
    def forward(self, image_features, text_features, merged_df=None, indices=None):
            
        device = image_features.device
        batch_size, feature_dim = image_features.size()
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        logits_per_image = self.logit_scale * image_features @ text_features.t()

        logits_per_text = logits_per_image.T
        if merged_df is not None:
            compare_matrix = merged_df.iloc[indices, 2:].to_numpy()
            vector_similarity_matrix = np.ones((compare_matrix.shape[0], compare_matrix.shape[0]), dtype=np.int32)
            comparison = (compare_matrix[:, None, :] == compare_matrix[None, :, :]).all(axis=2)
            vector_similarity_matrix[comparison] = 0
            np.fill_diagonal(vector_similarity_matrix, 1)
            vector_similarity_matrix = torch.from_numpy(vector_similarity_matrix).bool().to(device)
            masked_logits_per_image = logits_per_image.masked_fill(~vector_similarity_matrix, float('-inf'))
            masked_logits_per_text = logits_per_text.masked_fill(~vector_similarity_matrix.T, float('-inf'))
            loss = (F.cross_entropy(masked_logits_per_image, labels) + F.cross_entropy(masked_logits_per_text, labels)) / 2
        else:
            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return loss

class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.Linear(bottleneck_dim, dim)

        nn.init.kaiming_normal_(self.down.weight)
        nn.init.kaiming_normal_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

    
class CLIPLossACE_HGAT(nn.Module):
    def __init__(self, args, logit_scale, in_channels):
        super(CLIPLossACE_HGAT, self).__init__()
        set_random_seed(args.seed)
        self.args = args
        self.img_edge_adapter = ResidualAdapter(in_channels, args.hidden_features)
        self.text_edge_adapter = ResidualAdapter(in_channels, args.hidden_features)
        self.img_node_adapter = ResidualAdapter(in_channels, args.hidden_features)
        self.text_node_adapter = ResidualAdapter(in_channels, args.hidden_features)

        if args.learnable_logit_scale:
            self.logit_scale = nn.Parameter(logit_scale.clone().detach())
        else:
            self.register_buffer('logit_scale', logit_scale.clone().detach())
        
    def apply_ace_hgat(self, features, attn_weights, encoder="img"):
        
        if encoder =="img":
            edge_adapter = self.img_edge_adapter
            node_adapter = self.img_node_adapter
        elif encoder == 'text':
            edge_adapter = self.text_edge_adapter
            node_adapter = self.text_node_adapter
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
        topk_vals, topk_indices = torch.topk(sim, k=self.args.topk, dim=-1)
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
    
    def forward(self, clip_model, images, texts, merged_df=None, indices=None):
        
        device = images.device
        clip_model.visual.trunk.global_pool = ''
        image_features, img_attn_scores = clip_model.visual.trunk.get_attn_scores(images)
        image_features = F.normalize(clip_model.visual.head(image_features), dim=-1)
        text_features, text_attn_scores = clip_model.encode_text(texts, normalize=True, output_attentions=True, output_tokens=True)
        img_attn_scores = img_attn_scores.mean(dim=1) # [B, 197, 197]
        img_attn_weights = img_attn_scores[:, 0, 1:] # relationship between CLS token and patch embeddings [B, 196]
        
        text_attn_scores = text_attn_scores[-1].mean(dim=1) # [B, 256, 256]
        text_attn_weights = text_attn_scores[:, 0, 1:] # relationship between global token and other token embeddings [B, 255]
        

        if self.args.apply_gnn_encoders == 'vision':
            image_features = self.apply_ace_hgat(image_features, img_attn_weights, encoder="img")
            image_features = F.normalize(image_features, dim=-1)

            logits_per_image = self.logit_scale * image_features[:, 0] @ text_features[:, 0].t()
            logits_per_text = logits_per_image.T

        elif self.args.apply_gnn_encoders == 'text':
            text_features = self.apply_ace_hgat(text_features, text_attn_weights, encoder="text")
            text_features = F.normalize(text_features, dim=-1)

            logits_per_image = self.logit_scale * image_features[:, 0] @ text_features[:, 0].t()
            logits_per_text = logits_per_image.T

        elif self.args.apply_gnn_encoders == 'both':
            image_features = self.apply_ace_hgat(image_features, img_attn_weights, encoder="img")
            image_features = F.normalize(image_features, dim=-1)

            text_features = self.apply_ace_hgat(text_features, text_attn_weights, encoder="text")
            text_features = F.normalize(text_features, dim=-1)

            logits_per_image = self.logit_scale * image_features[:, 0] @ text_features[:, 0].t()
            logits_per_text = logits_per_image.T

        labels = torch.arange(image_features.shape[0], device=device, dtype=torch.long)

        if logits_per_image.isnan().sum() > 0:
            raise ValueError('NaN value in logits_per_image')
    
        if merged_df is not None: # Label-Guided InfoNCE loss
            compare_matrix = merged_df.iloc[indices, 2:].to_numpy()
            vector_similarity_matrix = np.ones((compare_matrix.shape[0], compare_matrix.shape[0]), dtype=np.int32)
            comparison = (compare_matrix[:, None, :] == compare_matrix[None, :, :]).all(axis=2)
            vector_similarity_matrix[comparison] = 0
            np.fill_diagonal(vector_similarity_matrix, 1)
            vector_similarity_matrix = torch.from_numpy(vector_similarity_matrix).bool().to(device)
            masked_logits_per_image = logits_per_image.masked_fill(~vector_similarity_matrix, float('-inf'))
            masked_logits_per_text = logits_per_text.masked_fill(~vector_similarity_matrix.T, float('-inf'))
            loss = (F.cross_entropy(masked_logits_per_image, labels) + F.cross_entropy(masked_logits_per_text, labels)) / 2
        else:
            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return loss