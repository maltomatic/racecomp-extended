import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import warnings

import torch, torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
import math
import time
import copy
from datetime import datetime
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

trim_dataset = True
if(trim_dataset):
    from loaders.load_redux import FairFaceDataset_Trim as FairFaceDataset
    print("Using FairFaceDataset_Trim from load_redux.py")
    from loaders.load_redux import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path
else:
    from loaders.load import FairFaceDataset
    print("Using FairFaceDataset from load.py")
    from loaders.load import race_weighted_sampler
    from loaders.load import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path

from models.cnn_56_interNet import Resnet_Interpolate_Upscaler as InterNet
from models.cnn_56_bridged_unet import Resnet_upscaler as UResNet
from models.cnn_abridged_unet import Resnet_upscaler_trim as TrimResNet
from models.vit_56 import VitUpscaler as VitNet

from toolkit.VGGPerceptionLoss import PerceptualLossVGG19
from toolkit.debugs import denormalize_imagenet
from toolkit.criteria import psnr, ssim_simple

B = 32
C = 3
H_l = W_l = 56
H_h = W_h = 224

#################### configs #################### 
TRAINING = False
debug = False
resume = False
training_comment = "Interpolation-net formal train"

model_idx = 1
# idx:
    # 0 - VitNet
    # 1 - InterNet
    # 2 - UResNet
    # 3 - TrimResNet
# train_list = ["All", "East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
train_list = ["All"]
use_percep = True
perc = 0.1
use_ssim = False
microbatches = 4 # 1 for no microbatching, n for n-step microbatching, max 8 recommended to avoid gradient explosion
sz = 56 # or 56
epoch_stages = (2, 1, 0, 0) if sz == 112 else (2, 2, 0, 0)

under_represented_ratio = 0.05
tgt_race = "All"
test_stage = 2
test_epoch = 4

config_str = f"size{sz}_mb{microbatches}_percep{int(use_percep)}_ssim{int(use_ssim)}"
#################################################
model_names = ["VitNet", "InterNet", "UResNet", "TrimResNet"]

Modelnet = VitNet(base_model = "fastvit_ma36", factor = 2,
                  input_shape=(3, sz, sz), out_shape = (3, 224, 224)) if model_idx == 0 else \
           InterNet(px_shuffle=False, px_shuffle_interpolate=True,
                    px_buffer = True, px_out=False, downsampler = True,
                    input_shape=(3, sz, sz)) if model_idx == 1 else \
           UResNet(px_shuffle = True, input_shape = (3, sz, sz)) if model_idx == 2 else \
           TrimResNet(output_size=(224,224), px_shuffle = True)
model_type = model_names[model_idx]
desc_path = f"{model_type}/{config_str}/"
desc = f"trained_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

train_stages = (["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]) if sz == 112\
    else (["enc3"], ["enc3","enc2"], ["enc1","enc2","enc3"])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(False)

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

def make_param_groups(model, base_lr=3e-4, dec_mult=1.0, enc_mult=0.5):
    enc_names = {"entry","enc1","enc2","enc3","enc4"}
    enc_ids = set()
    enc_params, dec_params = [], []
    for name in enc_names:
        m = getattr(model, name, None)
        if m is None: continue
        for p in m.parameters():
            enc_params.append(p); enc_ids.add(id(p))
    for p in model.parameters():
        if id(p) not in enc_ids:
            dec_params.append(p)
    return [
        {"params": dec_params, "lr": base_lr*dec_mult},
        {"params": enc_params, "lr": base_lr*enc_mult},
    ]

def accumulate_by_race(bucket, race, loss, psnr_val, ssim_val):
    b = bucket.setdefault(race, {"loss": [], "psnr": [], "ssim": []})
    b["loss"].append(loss); b["psnr"].append(psnr_val); b["ssim"].append(ssim_val)

def train(model, 
          train_loader, 
          val_loader, 
          stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"], ["entry","enc1","enc2","enc3","enc4"]),
          epochs_per_stage=(2, 2, 2, 4),
          lr=0.003,
          out_dir="checkpoints",
          microbatch_steps = 8,
          use_perceptual = True,
          use_ssim = True,
          perc = 0.1,
          resume = False):
    
    strikes = 0

    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=True)

    criterion = nn.L1Loss()
    perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
    perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
    lambda_perc = perc
    perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
    perceptual_criterion.eval()

    best_val_ssim = -1.0
    best_val_loss = float('inf')
    global_epoch = 0
    start_epoch = 0
    resume_batch = 0

    if(resume):
        svpt = torch.load('savepoints/ckpt_s0_e1_b188.pt', weights_only=False)

        print("Verifying checkpoint keys: ", svpt.keys())
        model.load_state_dict(svpt['model_state'])
        print("Resumed model state from savepoint.")
        print(f"Please ensure testing minority is {svpt["race"]}")
    try:
        for stage_idx, layer_list in enumerate(stages):
            # Unfreeze policy
            for name in ["entry","enc1","enc2","enc3","enc4"]:
                m = getattr(model, name, None)
                if m is not None:
                    for param in m.parameters():
                        param.requires_grad = False
                for name in layer_list:
                    m = getattr(model, name, None)
                    if m is not None:
                        for param in m.parameters():
                            param.requires_grad = True  
                # decoder always trainable
                enc_names = {"entry","enc1","enc2","enc3","enc4"}
                for name, m in model.named_children():
                    if name not in enc_names:
                        for param in m.parameters():
                            param.requires_grad = True
            total_epochs = epochs_per_stage[stage_idx]
            optimizer = torch.optim.AdamW(make_param_groups(model, lr, dec_mult=1.0, enc_mult=0.25), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
            
            if(resume and stage_idx == svpt["stage"]):
                start_epoch = svpt['epoch'] - 1
                global_epoch += start_epoch
                resume_batch = svpt['batch']
                tr = svpt['losses']
                print(f"Resumed optimizer and scheduler state from savepoint at stage {svpt["stage"]}, epoch {global_epoch}, batch {resume_batch}.")
            elif (resume and stage_idx < svpt["stage"]):
                print(f"Skipping stage {stage_idx}, resume savepoint at stage {svpt['stage']}.")
                global_epoch += total_epochs
                continue  # skip earlier stages if resuming

            print(f"\n=== Stage {stage_idx+1}/{len(stages)} | Unfrozen: {layer_list} ===")
            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                file.write(f"\n=== Stage {stage_idx+1}/{len(stages)} | Unfrozen: {layer_list} ===\n")

            for e in range(start_epoch, total_epochs):
                global_epoch += 1
                print(f"\n--- Epoch {e+1}/{total_epochs} time: {datetime.now().strftime("%H:%M:%S")}---")
                
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"\n--- Epoch {global_epoch} || time: {datetime.now().strftime("%H:%M:%S")} --- \n")
                model.train()
                if(not resume):
                    tr = defaultdict(float)
                optimizer.zero_grad(set_to_none=True)
                n_batches = 0

                if(debug):
                    print("Debug: Skipping training loop in debug mode.")
                    continue

                for X_img, Y_img, labels, label_str in train_loader:  # LR, HR
                    if(n_batches < resume_batch and resume):
                        n_batches += 1
                        continue
                    elif(resume):
                        resume = False
                        scheduler.load_state_dict(svpt['scheduler_state'])
                        # optimizer.load_state_dict(svpt['optim_state'])
                        # scaler.load_state_dict(svpt['scaler_state'])
                        print(f"Skipped to batch {n_batches} ({resume_batch}) to resume from savepoint.")
                        #verify
                        print("==================================================================")
                        print(f"Currently at stage {stage_idx}, epoch {e}, batch {n_batches}.")
                        print(f"Start epoch {start_epoch}, global epoch {global_epoch}, total epoch {total_epochs}.")

                    Y_img = Y_img.to(device).float()
                    X_img = X_img.to(device).float()

                    with torch.amp.autocast(device_type, enabled=True):
                        # print("Debug: Input shapes:", X_img.shape, Y_img.shape)
                        pred = model(X_img)
                        if not torch.isfinite(pred).all():
                            print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                                file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                            optimizer.zero_grad(set_to_none=True)
                            n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                            # 3 strikes
                            torch.autograd.set_detect_anomaly(strikes < 2)
                            strikes += 1
                            continue
                        pixel_loss = criterion(pred, Y_img)
                        if use_perceptual:
                            with torch.amp.autocast(device_type, enabled = True):
                                perc_loss = perceptual_criterion(pred, Y_img)
                            # loss = pixel_loss + lambda_perc * perc_loss
                            loss = perc_loss + lambda_perc * pixel_loss
                        else:
                            loss = pixel_loss
                        loss = loss / microbatch_steps  # Scale loss for gradient accumulation
                        if not torch.isfinite(loss).all():
                            print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                                file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                            optimizer.zero_grad(set_to_none=True)
                            n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                            # 3 strikes
                            torch.autograd.set_detect_anomaly(strikes < 2)
                            strikes += 1
                            continue

                    scaler.scale(loss).backward()

                    if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        scheduler.step()
                    
                    if n_batches % 120 == 0 and use_perceptual:
                        print(f"pixel={pixel_loss.item():.4f}  perc={perc_loss.item():.4f}  tot={(loss.item() * microbatch_steps):.4f}")

                    pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
                    targ_vis = imagenet_denorm(Y_img).clamp(0.0, 1.0)
                    tr["loss"] += loss.item() * microbatch_steps # unscale the loss
                    tr["psnr"] += psnr(pred_vis, targ_vis).mean().item()
                    tr["ssim"] += ssim_simple(pred_vis, targ_vis).mean().item()

                    n_batches += 1
                    if(n_batches % 200 == 1):
                        print(f"Batch {n_batches:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                            f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}")
                        print(f"At step {n_batches + 1}, Learning rate {scheduler.get_last_lr()[0]}")
                    if(n_batches % 2000 == 1):
                        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                            file.write(f"Batch {n_batches:03d} at time {datetime.now().strftime("%H:%M:%S")} | train: loss {tr['loss']/n_batches:.4f}  "
                                    f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}\n")

                print(f"Epoch {global_epoch:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                    f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"Epoch {global_epoch:03d} | train: loss {tr['loss']/n_batches:.4f}  "
                            f"PSNR {tr['psnr']/n_batches:.2f}  SSIM {tr['ssim']/n_batches:.4f}\n")
                
                model.eval()
                print(f"==Validation: || time: {datetime.now().strftime("%H:%M:%S")} ==")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"==Validation: || time: {datetime.now().strftime("%H:%M:%S")} ==\n")
                v = defaultdict(float); n_val = 0; race_bucket = {}
                with torch.no_grad():
                    for X_img, Y_img, labels, label_str in val_loader:
                        Y_img = Y_img.to(device).float()
                        X_img = X_img.to(device).float()

                        pred = model(X_img)
                        val_pixel = criterion(pred, Y_img)
                        if use_perceptual:
                            with torch.amp.autocast(device_type, enabled = True):
                                val_perc = perceptual_criterion(pred, Y_img)
                            # val_loss = val_pixel + lambda_perc * val_perc
                            val_loss = val_perc + lambda_perc * val_pixel
                        else:
                            val_loss = val_pixel

                        pred_vis = imagenet_denorm(pred).clamp(0.0, 1.0)
                        targ_vis = imagenet_denorm(Y_img).clamp(0.0, 1.0)
                        v_psnr = psnr(pred_vis, targ_vis).mean().item()
                        v_ssim = ssim_simple(pred_vis, targ_vis).mean().item()

                        v["loss"] += val_loss.item()
                        v["psnr"] += v_psnr
                        v["ssim"] += v_ssim
                        n_val += 1

                        for r in label_str:
                            accumulate_by_race(race_bucket, r, val_loss.item(), v_psnr, v_ssim)

                val_loss = v["loss"]/n_val
                val_psnr = v["psnr"]/n_val
                val_ssim = v["ssim"]/n_val
                print(f"           val:   loss {val_loss:.4f}  PSNR {val_psnr:.2f}  SSIM {val_ssim:.4f}")

                race_summary = {}
                for k, v in race_bucket.items():
                    race_summary[k] = {m: float(np.mean(vals)) if len(vals)>0 else float('nan') for m, vals in v.items()}
                
                if race_summary:
                    print("           per-race (val):",
                        "  ".join([f"{k}: SSIM {vals['ssim']:.3f}, PSNR {vals['psnr']:.2f}"
                                    for k, vals in race_summary.items()]))
                    with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                        file.write("per-race (val): " + "  ".join([f"{k}: SSIM {vals['ssim']:.3f}, PSNR {vals['psnr']:.2f}" for k, vals in race_summary.items()]))

                if use_ssim:
                    if val_ssim > best_val_ssim:
                        best_val_ssim = val_ssim
                        ckpt = {
                            "model": model.state_dict(),
                            "val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim,
                            "race_summary": race_summary,
                            "stage": stage_idx+1, "epoch": global_epoch
                        }
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir, exist_ok=True)
                        path = os.path.join(out_dir, f"best_stage{stage_idx+1}_epoch{global_epoch}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt")
                        torch.save(ckpt, path)
                        print(f"Saved best checkpoint → {path} (SSIM={val_ssim:.4f})")
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt = {
                            "model": model.state_dict(),
                            "val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim,
                            "race_summary": race_summary,
                            "stage": stage_idx+1, "epoch": global_epoch
                        }
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir, exist_ok=True)
                        path = os.path.join(out_dir, f"best_stage{stage_idx+1}_epoch{global_epoch}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt")
                        torch.save(ckpt, path)
                        print(f"Saved best checkpoint → {path} (SSIM={val_ssim:.4f})")
            del optimizer
            del scheduler
    except:
        print("Save checkpoint before quitting? (y/n)")
        if(input().lower().startswith('y')):
            if not os.path.exists('savepoints'):
                os.makedirs('savepoints', exist_ok=True)
            ckpt_epoch = global_epoch
            ckpt_stage = stage_idx
            ckpt_batch = n_batches
            ckpt_race = minority
            ckpt_model_state = model.state_dict()
            ckpt_optim_state = optimizer.state_dict()
            ckpt_scheduler_state = scheduler.state_dict()
            ckpt_scaler_state = scaler.state_dict()
            ckpt_tr = tr
            torch.save({
                "model_state": ckpt_model_state,
                "optim_state": ckpt_optim_state,
                "scheduler_state": ckpt_scheduler_state,
                "scaler_state": ckpt_scaler_state,
                "epoch": ckpt_epoch,
                "batch": ckpt_batch,
                "stage": ckpt_stage,
                "race": ckpt_race,
                "losses": ckpt_tr
            }, f'savepoints/ckpt_s{ckpt_stage}_e{ckpt_epoch}_b{ckpt_batch}.pt')
        raise

        

race_weights = {
    "East Asian": 0.2,
    "Indian": 1.0,
    "Black": 1.0,
    "White": 1.0,
    "Middle Eastern": 1.0,
    "Latino_Hispanic": 1.0,
    "Southeast Asian": 1.0,
}

if __name__ == "__main__":
    if(model_type == "TrimResNet"):
        print("TrimResNet model selected, ensure input size is 112x112 due to internal structure.")
        assert sz == 112, "TrimResNet only supports input size of 112x112."
    print("Using device: ", device)
    print(f"Logging to logs/training/{desc_path}{desc}.txt with configs:\n - Model: {model_type}, Image Size: {sz}, Batch size: {B}, Microbatch steps: {microbatches}, Use Perceptual Loss: {use_percep}, Use SSIM: {use_ssim}")
    if(torch.cuda.is_available()):
        print(f"GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    if TRAINING:
        if not os.path.exists(f"./logs/training/{desc_path}"):
            os.makedirs(f"./logs/training/{desc_path}", exist_ok=True)
        with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
            file.write(f"\n\n=== {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : {training_comment} ===\n")
            file.write(f"Training on GPU ID: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
            file.write(f"\nConfigs - Model: {model_type}, Image Size: {sz}, Batch size: {B}, Microbatch steps: {microbatches}, Use Perceptual Loss: {use_percep}, Use SSIM: {use_ssim}")

        if(not trim_dataset):
           train_dataset = FairFaceDataset(train_image_path, train_label_path, lr_size=(sz, sz))
           train_loader = None
        val_dataset = FairFaceDataset(val_image_path, val_label_path, lr_size=(sz, sz))
        val_loader = DataLoader(val_dataset, batch_size=B, shuffle=trim_dataset, num_workers=8, pin_memory=True)

        for minority in train_list:
            model = copy.deepcopy(Modelnet)
            for par1, par2 in zip(model.parameters(), Modelnet.parameters()):
                assert torch.allclose(par1, par2), "Model copy failed!"
            model.to(device)
            print(f"Created copy of {model_type} initialized for training.")

            if(not trim_dataset):
                rm = None
                if minority == "All":
                    rm = {r: 1.0 for r in race_weights.keys()}
                else:
                    rm = {r: (under_represented_ratio if r == minority else 1.0) for r in race_weights.keys()}
                sampler = race_weighted_sampler(train_dataset, rm, num_samples=len(train_dataset), seed=42)
                train_loader = DataLoader(train_dataset, batch_size=B, shuffle=False, sampler=sampler, num_workers=8, pin_memory=True)
            else:
                train_dataset = FairFaceDataset(train_image_path, train_label_path, lr_size = (sz, sz), minority=minority, minority_weight=under_represented_ratio)
                train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True)

            # print("Number of training samples: ", len(train_dataset))
            # print("Number of validation samples: ", len(val_dataset))
            print(f"\n=== Training with minority: {minority} ===")
            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                file.write(f"\n=== Training with minority: {minority} ===\n")
                # file.write(f"\nNumber of training samples: {len(train_dataset)}")
                # file.write(f"\nNumber of validation samples: {len(val_dataset)}\n")
            
            print("Bootstrapping data loaders")
            for images, src, labels, label_str in train_loader:
                print("Batch of testing images shape: ", images.shape)
                print("Batch of source images shape: ", src.shape)
                print("Batch of labels shape: ", labels.shape)
                print("Batch of label strings: ", len(label_str))
                break
            
            torch.autograd.set_detect_anomaly(False)
            train(
                model,
                train_loader,
                val_loader,
                stages=train_stages,
                epochs_per_stage= epoch_stages, #(2, 2, 3, 1),
                lr=3e-4,
                out_dir=f"checkpoints/{model_type}/config_{config_str}/minority_{minority.replace(" ","_")}",
                use_perceptual= use_percep,
                use_ssim= use_ssim,
                microbatch_steps = microbatches,
                perc = perc,
                resume = resume
            )
            resume = False

            print(f"Finished training for minority: {minority}, releasing model from GPU.\n\n")
            with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                file.write(f"\nFinished training for minority: {minority}.\n\n")

            del model
            torch.cuda.empty_cache()
    else:
        model = Modelnet.to(device)
        print("Load model from checkpoint for inference/testing")
        ckpt_path = f"checkpoints/{model_type}/config_{config_str}/minority_{tgt_race.replace(" ", "_")}/best_stage{test_stage}_epoch{test_epoch}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict = False)
        model.eval()
        print(f"Loaded model from {ckpt_path} | val SSIM: {ckpt['val_ssim']:.4f}")
        model.to(device)

        #load test sample image
        for i in range(1, 5):
            testpath = f"test_56_{i}.png"
            # testpath = f"test_{sz}.png"
            img_file = f".//test_files//{testpath}"
            image = decode_image(img_file, mode = "RGB")
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((112, 112)),
                torchvision.transforms.ConvertImageDtype(torch.float),
                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            input_img = transform(image).unsqueeze(0).to(device)  # add batch dimension
            with torch.no_grad():
                pred = model(input_img)
                # save output image
            output_img = denormalize_imagenet(pred.squeeze(0).cpu()).permute(1, 2, 0).numpy()
            output_pil = Image.fromarray(output_img)
            output_pil.save(f"test_files/outputs/{model_type}_output_{testpath}.png")
            print(f"Saved output image to test_files/outputs/{model_type}_output_{testpath}.png")
