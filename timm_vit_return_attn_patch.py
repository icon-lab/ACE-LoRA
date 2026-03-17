import torch
import timm.models.vision_transformer as vit

def patch_timm_vit_return_attn_scores():
    _orig_attn_forward = vit.Attention.forward

    def attn_forward_patched(self, x, return_attn_scores = False):
        if not return_attn_scores:
            return _orig_attn_forward(self, x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn_scores = q @ k.transpose(-2, -1)
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return (x, attn_scores)

    vit.Attention.forward = attn_forward_patched

    # Patch Block.forward
    _orig_block_forward = vit.Block.forward

    def block_forward_patched(self, x, return_attn_scores= False):
        if not return_attn_scores:
            return _orig_block_forward(self, x)

        out, attn_scores = self.attn(self.norm1(x), return_attn_scores=True)
        x = x + self.drop_path1(self.ls1(out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return (x, attn_scores)

    vit.Block.forward = block_forward_patched

    def get_attn_scores(self, x, pre_logits: bool = False):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        depth = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            if i == (depth - 1):
                x, attn_scores = blk(x, return_attn_scores=True)
            else:
                x = blk(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)

        if not pre_logits:
            x = self.head(x)

        return (x, attn_scores)

    vit.VisionTransformer.get_attn_scores = get_attn_scores
