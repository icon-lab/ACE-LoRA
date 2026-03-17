import torch
from utils import *
import os
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, save_lora
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from loss import * 
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
from timm_vit_return_attn_patch import patch_timm_vit_return_attn_scores
from bert_modeling_bert_self_attn_patch import patch_bert_self_attn

def run_model(args, clip_model, merged_df, tokenizer, logit_scale, train_loader, preprocess):
    
    if args.save_path == None:
        raise ValueError('args.save_path cannot be None')
    now = datetime.now()
    foldername_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = f'{args.save_path}/seed{args.seed}/{foldername_time}'
    os.makedirs(save_dir, exist_ok=True)

    patch_timm_vit_return_attn_scores()
    patch_bert_self_attn()
    # save hyperparameter values 
    with open(os.path.join(save_dir, 'model_hyperparameters.txt'),'w') as file:
        for key, value in vars(args).items():
            file.write(f"{key} : {value} \n")

    list_lora_layers = apply_lora(args, clip_model)
    mark_only_lora_as_trainable(clip_model)
    
    clip_model = clip_model.to(args.device)
    num_epochs = args.num_epochs

    feature_dim = 512

    if args.loss_type == 'clip_loss':
        loss_fn = CLIPLoss(args, logit_scale).to(device=args.device)
    elif args.loss_type == 'clip_loss_ace_hgnn':
        loss_fn = CLIPLossACE_HGAT(args, logit_scale, feature_dim).to(device=args.device)
    else:
        raise ValueError(f'Invalid loss type is given: {args.loss_type}')

    learnable_params = get_lora_parameters(clip_model) + list(loss_fn.parameters())
    print("Number of learnable params: ", sum(p.numel() for p in learnable_params))
    optimizer = torch.optim.AdamW(learnable_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader)*num_epochs, num_cycles=0.5)

    best_epoch_loss = -1
    best_acc_chexpert = -1
    best_auc_chexpert = -1
    best_acc_rsna = -1
    best_auc_rsna = -1
    best_acc_siim = -1
    best_auc_siim = -1
    best_auc_chest_xray_14 = -1

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        clip_model.train()
        tot_samples = 0
        loss_epoch = 0.
        for i, (images, target, index) in enumerate(tqdm(train_loader)):
            indices = index.tolist()
            images = images.to(args.device)
            if args.loss_type == 'clip_loss_ace_hgnn':
                texts = tokenizer(target, context_length = args.context_length).to(args.device)
                loss = loss_fn(clip_model, images, texts, merged_df, indices)
            else:
                texts = tokenizer(target, context_length = args.context_length).to(args.device)
                text_features = clip_model.encode_text(texts, normalize=True) # Note that this normalization is L2 Norm, which is different than the frobenius norm (default option for torch.norm)
                image_features = clip_model.encode_image(images, normalize=True)
                loss = loss_fn(image_features, text_features, merged_df, indices)

            loss_epoch += loss.item() * images.shape[0]
            tot_samples += images.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
                
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Loss: {:.4f}'.format(current_lr, loss_epoch))
        
        if best_epoch_loss == -1 or loss_epoch < best_epoch_loss:
            print(f'Saving best epoch loss: {loss_epoch}')
            print(f'current logit_scale: {loss_fn.logit_scale.item()}')
            logit_scale = loss_fn.logit_scale
            best_epoch_loss = loss_epoch
            msg = f'best_epoch_{epoch + 1}'
            save_lora(args, list_lora_layers, loss_fn, msg, save_dir)
            if args.eval:
                clip_model.eval()
                acc, auc = zero_shot_chexpert_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=False)
                if acc > best_acc_chexpert:
                    best_acc_chexpert = acc
                if auc > best_auc_chexpert:
                    best_auc_chexpert = auc
                with open(os.path.join(save_dir, 'chexpert5x200_eval.txt'), 'a') as file:
                    file.write(f'Chexpert 5x200: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f}, Best Accuracy: {best_acc_chexpert:.4f}, Best AUROC: {best_auc_chexpert:.4f} \n')
                
                acc, auc = zero_shot_rsna_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=False)
                if acc > best_acc_rsna:
                    best_acc_rsna = acc
                if auc > best_auc_rsna:
                    best_auc_rsna = auc
                with open(os.path.join(save_dir, 'RSNA_eval.txt'), 'a') as file:
                    file.write(f'RSNA: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f}, Best Accuracy: {best_acc_rsna:.4f}, Best AUROC: {best_auc_rsna:.4f} \n')

                acc, auc = zero_shot_siim_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=False)
                if acc > best_acc_siim:
                    best_acc_siim = acc
                if auc > best_auc_siim:
                    best_auc_siim = auc
                with open(os.path.join(save_dir, 'SIIM_eval.txt'), 'a') as file:
                    file.write(f'SIIM: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f}, Best Accuracy: {best_acc_siim:.4f}, Best AUROC: {best_auc_siim:.4f} \n')

                # used for validation
                auc = zero_shot_chest_xray_14_eval(args, clip_model, tokenizer, loss_fn, preprocess, trained_on_multi_gpu=False)
                if auc > best_auc_chest_xray_14:
                    best_auc_chest_xray_14 = auc
                with open(os.path.join(save_dir, 'Chest_xray_14_eval.txt'), 'a') as file:
                    file.write(f'Chest X-ray 14: Epoch: {epoch + 1}, AUROC: {auc:.4f}, Best AUROC: {best_auc_chest_xray_14:.4f} \n')

        with open(os.path.join(save_dir, 'loss.txt'), 'a') as file:
            file.write(f'Epoch: {epoch + 1}, LR : {current_lr:.6f}, loss : {loss_epoch:.4f}, Best loss : {best_epoch_loss:.4f}, logit_scale: {loss_fn.logit_scale.item():.4f} \n')
            
    return

def run_model_multi_gpu(args, clip_model, merged_df, tokenizer, logit_scale, train_loader, preprocess):
    
    if args.save_path == None:
        raise ValueError('args.save_path cannot be None')
    
    patch_timm_vit_return_attn_scores()
    patch_bert_self_attn()
    
    if args.rank == 0:
        now = datetime.now()
        foldername_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = f'{args.save_path}/seed{args.seed}/{foldername_time}'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'model_hyperparameters.txt'),'w') as file:
            for key, value in vars(args).items():
                file.write(f"{key} : {value} \n")

    list_lora_layers = apply_lora(args, clip_model)
    mark_only_lora_as_trainable(clip_model)
    
    clip_model = clip_model.to(args.rank)
    clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
    clip_model = DDP(clip_model, device_ids=[args.rank])

    num_epochs = args.num_epochs
    feature_dim = 512

    if args.loss_type == 'clip_loss':
        loss_fn = CLIPLoss(args, logit_scale).to(device=args.rank)
    elif args.loss_type == 'clip_loss_ace_hgnn':
        loss_fn = CLIPLossACE_HGAT(args, logit_scale, feature_dim).to(device=args.rank)
    else:
        raise ValueError(f'Invalid loss type is given: {args.loss_type}')
    
    learnable_params = get_lora_parameters(clip_model) + list(loss_fn.parameters())
    print("Number of learnable params: ", sum(p.numel() for p in learnable_params))
    optimizer = torch.optim.AdamW(learnable_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader)*num_epochs, num_cycles=0.5)
    
    best_epoch_loss = -1
    best_acc_chexpert = -1
    best_auc_chexpert = -1
    best_acc_rsna = -1
    best_auc_rsna = -1
    best_acc_siim = -1
    best_auc_siim = -1
    best_auc_chest_xray_14 = -1
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        clip_model.module.train()
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target, index) in enumerate(tqdm(train_loader)):
            indices = index.tolist()
            images = images.to(args.rank)

            if args.loss_type == 'clip_loss_ace_hgnn':
                texts = tokenizer(target, context_length = args.context_length).to(args.rank)
                loss = loss_fn(clip_model.module, images, texts, merged_df, indices)
            else:
                texts = tokenizer(target, context_length = args.context_length).to(args.rank)
                text_features = clip_model.module.encode_text(texts, normalize=True) # Note that this normalization is L2 Norm, which is different than the frobenius norm (default option for torch.norm)
                image_features = clip_model.module.encode_image(images, normalize=True)
                loss = loss_fn(image_features, text_features, merged_df, indices)

            loss_epoch += loss.item() * images.shape[0]
            tot_samples += images.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.rank == 0: 
                scheduler.step()
            lr = scheduler.get_last_lr()[0]
            torch.distributed.broadcast(torch.tensor(lr, device=args.rank), 0)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        dist.barrier()

        if args.rank == 0:
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Loss: {:.4f}'.format(current_lr, loss_epoch))
            
            if best_epoch_loss == -1 or loss_epoch < best_epoch_loss:
                print(f'Saving best epoch loss: {loss_epoch}')
                print(f'current logit_scale: {loss_fn.logit_scale.item()}')
                logit_scale = loss_fn.logit_scale
                best_epoch_loss = loss_epoch
                msg = f'best_epoch_{epoch + 1}'
                save_lora(args, list_lora_layers, loss_fn, msg, save_dir)

                if args.eval:
                    clip_model.module.eval()
                    acc, auc = zero_shot_chexpert_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=True)
                    if acc > best_acc_chexpert:
                        best_acc_chexpert = acc
                    if auc > best_auc_chexpert:
                        best_auc_chexpert = auc
                    with open(os.path.join(save_dir, 'chexpert5x200_eval.txt'), 'a') as file:
                        file.write(f'Chexpert 5x200: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f}, Best Accuracy: {best_acc_chexpert:.4f}, Best AUROC: {best_auc_chexpert:.4f} \n')

                    acc, auc = zero_shot_rsna_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=True)
                    if acc > best_acc_rsna:
                        best_acc_rsna = acc
                    if auc > best_auc_rsna:
                        best_auc_rsna = auc
                    with open(os.path.join(save_dir, 'RSNA_eval.txt'), 'a') as file:
                        file.write(f'RSNA: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f}, Best Accuracy: {best_acc_rsna:.4f}, Best AUROC: {best_auc_rsna:.4f} \n')
                    
                    acc, auc = zero_shot_siim_eval(args, clip_model, tokenizer, loss_fn, preprocess, logit_scale, trained_on_multi_gpu=True)
                    if acc > best_acc_siim:
                        best_acc_siim = acc
                    if auc > best_auc_siim:
                        best_auc_siim = auc
                    with open(os.path.join(save_dir, 'SIIM_eval.txt'), 'a') as file:
                        file.write(f'SIIM: Epoch: {epoch + 1}, Accuracy: {acc:.4f}, AUROC: {auc:.4f},  Best Accuracy: {best_acc_siim:.4f}, Best AUROC: {best_auc_siim:.4f} \n')

                    # used for validation
                    auc = zero_shot_chest_xray_14_eval(args, clip_model, tokenizer, loss_fn, preprocess, trained_on_multi_gpu=True)
                    if auc > best_auc_chest_xray_14:
                        best_auc_chest_xray_14 = auc
                    with open(os.path.join(save_dir, 'Chest_xray_14_eval.txt'), 'a') as file:
                        file.write(f'Chest X-ray 14: Epoch: {epoch + 1}, AUROC: {auc:.4f}, Best AUROC: {best_auc_chest_xray_14:.4f} \n')

            with open(os.path.join(save_dir, 'loss.txt'), 'a') as file:
                file.write(f'Epoch: {epoch + 1}, LR : {current_lr:.6f}, loss : {loss_epoch:.4f}, Best loss : {best_epoch_loss:.4f}, logit_scale: {loss_fn.logit_scale.item():.4f} \n')

        dist.barrier()

    return
    
            
