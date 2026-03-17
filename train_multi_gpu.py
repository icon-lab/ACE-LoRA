import torchvision.transforms as transforms
from dataset import *
from open_clip import create_model_from_pretrained, get_tokenizer
import pandas as pd
from utils import *
from run_utils_multi_gpu import *
from lora import run_model_multi_gpu
import os
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, DataLoader
from open_clip_patch import patch_encode_text

def worker_init_fn(worker_id):
    seed = get_arguments().seed
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

def main_worker(rank, world_size, args):

    setup_ddp(rank, world_size)
    set_random_seed(args.seed)
    args.rank = rank
    args.num_devices = world_size

    cache_dir = './.cache/hub'
    os.environ['HF_HOME'] = cache_dir
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    clip_model, preprocess = create_model_from_pretrained(model_name, device=args.rank, cache_dir=cache_dir)
    
    patch_encode_text()
    logit_scale = clip_model.state_dict()['logit_scale'].exp()
    tokenizer = get_tokenizer(model_name)
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
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed, rank=rank, num_replicas=world_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, pin_memory=True, sampler=train_sampler, worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(args.seed), persistent_workers=True)
    
    try:
        run_model_multi_gpu(args, clip_model, merged_df, tokenizer, logit_scale, train_loader, preprocess)
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    args = get_arguments()
    gpus = [0, 1, 2] # select GPUs
    world_size = len(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)