import random
import argparse  
import numpy as np 
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--loss_type', type=str, choices=['clip_loss', 'clip_loss_ace_hgnn'], default='clip_loss_ace_hgnn') 
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=4, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--save_path', default='./ACE-LoRA', help='path to save trainable params')
    parser.add_argument('--filename', default='ACE-LoRA_weights', help='file name to save the lora weights (.pt extension will be added)')
    parser.add_argument('--eval', default=True, action='store_true', help='perform evaluation')
    parser.add_argument('--context_length', default=256, help='context length for tokenizer')
    parser.add_argument('--csv_path', default='./mimic_cxr.csv', help='csv path for MIMIC-CXR dataset')
    parser.add_argument('--label_csv_path', default='mimic-cxr-chexpert_labeled_data.csv', help='chexpert labeled csv path for MIMIC-CXR dataset')
    parser.add_argument('--use_labels', default=True, help='use disease labels during training (Label-Guided InfoNCE loss)')
    parser.add_argument('--learnable_logit_scale', default=True)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--hidden_features', default=128, help='output dimension of linear projection in HGNN')
    parser.add_argument('--apply_gnn_encoders', type=str, choices=['text', 'vision', 'both'], default='both')
    args = parser.parse_args()

    return args
    

        
