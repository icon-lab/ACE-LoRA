from transformers.models.bert import modeling_bert
from open_clip import CustomTextCLIP
from open_clip.hf_model import HFTextEncoder
import torch.nn.functional as F
from torch import TensorType

def patch_encode_text():

    def encode_text_patched(self, text, normalize: bool = False, output_attentions = False, output_tokens = False):
        if output_attentions:
                features, attn_scores = self.text(text, output_attentions = output_attentions, output_tokens = output_tokens)
                features = F.normalize(features, dim=-1) if normalize else features
                return features, attn_scores
        else:
            features = self.text(text, output_attentions = output_attentions, output_tokens = output_tokens)
            return F.normalize(features, dim=-1) if normalize else features

    def HFText_encoder_patched(self, x: TensorType, output_attentions=False, output_tokens=False):
        self.output_tokens = output_tokens
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask, output_attentions=output_attentions)
        if self.output_tokens:
            tokens = self.proj(out[0])
            if output_attentions:
                return tokens, out[1]
            else:
                return tokens
        else:
            pooled_out = self.pooler(out, attn_mask)
            projected = self.proj(pooled_out)

            return projected

    CustomTextCLIP.encode_text = encode_text_patched
    HFTextEncoder.forward = HFText_encoder_patched


