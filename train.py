import torch
import torchvision.transforms as transforms
from dataset import *
from open_clip import create_model_from_pretrained, get_tokenizer
import pandas as pd
from utils import *
from run_utils import *
from lora import run_model
import os
from open_clip_patch import patch_encode_text

def worker_init_fn(worker_id):
    seed = get_arguments().seed
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

def main():

    # load config file
    args = get_arguments()
    set_random_seed(args.seed)
    cache_dir = './.cache/hub'
    os.environ['HF_HOME'] = cache_dir
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    clip_model, preprocess = create_model_from_pretrained(model_name, device=args.device, cache_dir=cache_dir)
    logit_scale = clip_model.state_dict()['logit_scale'].exp()
    tokenizer = get_tokenizer(model_name)
    
    patch_encode_text()
    df = pd.read_csv(args.csv_path)
    df_train, _ , _ = data_split(df)
    df = df_train
    if args.use_labels: # Label-guided InfoNCE loss
        df_study_id = df['path_base'].str.split('/').str[-1]
        df_labeled = pd.read_csv(args.label_csv_path)
        df_labeled.fillna(0, inplace=True)
        df_labeled = df_labeled.replace(-1, 0)
        df_labeled['study_id'] = df_labeled['study_id'].apply(lambda id: 's' + str(id))
        study_df = pd.DataFrame()
        study_df['study_id'] = df_study_id
        merged_df = df_labeled.merge(study_df, on='study_id', how='inner')
    else:
        merged_df = None

    train_tranform = transforms.Compose([
            transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    print("Preparing dataset.")

    train_dataset = MIMIC_CXR_Dataset(df, transform = train_tranform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)
    
    run_model(args, clip_model, merged_df, tokenizer, logit_scale, train_loader, preprocess)

if __name__ == '__main__':
    main() 
