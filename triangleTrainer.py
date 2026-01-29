import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import warnings

import torch, torchvision
import torch.nn as nn
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

from loaders.load_redux import FairFaceDataset_Trim as FairFaceDataset
from loaders.load_redux import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path

from toolkit.VGGPerceptionLoss import PerceptualLossVGG19
from toolkit.debugs import denormalize_imagenet
from toolkit.criteria import psnr, ssim_simple

from models.triangleNet import TriangleNet

B = 12
C = 3
H_l = W_l = 56
H_h = W_h = 224

#################### configs #################### 
TRAINING = True
debug = False
resume = False
training_comment = "Testing TriangleNet"

# train_list = ["All", "East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
train_list = ["All"]
use_percep = True
perc = 0.1
use_ssim = False
microbatches = 8 # 1 for no microbatching, n for n-step microbatching, max 8 recommended to avoid gradient explosion
sz = 56 # or 56
epoch_stages = (8, 10, 5, 3)
# stage 1: fine-tune teacher, L1 loss
# stage 2: train learner against teacher, L2 + slight VGG loss
# stage 3: train decoder against teacher, VGG + slight L1 loss
# stage 4: fine tune decoder against learner, VGG + slight L1 loss
decoder_stages = (2, 2, 1) # encoder unfreeze level when training decoder in s3

under_represented_ratio = 0.05
tgt_race = "All"
test_stage = 2
test_epoch = 4

config_str = f"size{sz}_mb{microbatches}_percep{int(use_percep)}"
#################################################

model = TriangleNet(px_shuffle=False, px_shuffle_interpolate=True, training = True)
model_type = "TriangleNet"
desc_path = f"{model_type}/{config_str}/"
desc = f"trained_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
log_path = f"logs/training/{desc_path}{desc}.txt"
ckpt_path = f"checkpoints/{desc_path}"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(False)

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

def savepoint(model, stage, epoch):
    ckpt = {
        "model": model.state_dict()
    }
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    path = os.path.join(ckpt_path, f"best_stage{stage}_epoch{epoch}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt")
    torch.save(ckpt, path)
    print(f"Saved best checkpoint at {path}")
    with open(log_path, "a") as file:
        file.write(f"Saved best checkpoint at {path}\n")

def stage_1_train(model, train_loader, val_loader = None, st1_lr = 3e-4, st2_lr = 1e-5, st1_epochs = 4, st2_epochs = 2, microbatch_steps = 4):
    # clear memory
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)
    strikes = 0
    # fine-tune teacher, L1 loss
    # only model.teacher, model.teacher_upscaler trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.teacher_upscaler.parameters():
        param.requires_grad = True
    # two-stage training: first upscaler, then both
    
    criterion = nn.L1Loss()
    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler(device_type, enabled=True)

    optimizer = torch.optim.AdamW(model.teacher_upscaler.parameters(), lr=st1_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, st1_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    print("======== Stage 1-1 ========")
    with open(log_path, "a") as file:
        file.write(f"======== Stage 1-1 ========\n")
    for epoch in range(st1_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for X_img, Y_img, _, _ in train_loader:
            if(debug and n_batches >= 1):
                break
            # bilinear upscale X_img to Y_img size
            X_img = F.interpolate(X_img, size=(H_h, W_h), mode='bilinear', align_corners=False).to(device).float()
            X_img = torch.clamp(X_img, 0.0, 1.0)
            pred = model(X_img, stage=1)
            if not torch.isfinite(pred).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            Y_img = Y_img.to(device).float()
            loss = criterion(pred, Y_img)
            if not torch.isfinite(loss).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            loss /= microbatch_steps
            epoch_loss += loss.item()
            batch_loss += loss.item()
            scaler.scale(loss).backward()
            if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                scheduler.step()
            n_batches += 1
            if(n_batches % 500 == 0):
                with open(log_path, "a") as file:
                    file.write(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}\n")
                print(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}")
                batch_loss = 0.0
        print(f"Stage 1-1 Epoch {epoch + 1}/{st1_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}")
        with open(log_path, "a") as file:
            file.write(f"Stage 1-1 Epoch {epoch + 1}/{st1_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}\n")
    del optimizer
    del scheduler
    
    for param in model.teacher.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(list(model.teacher.parameters()) + list(model.teacher_upscaler.parameters()), lr=st2_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, st2_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    print("======== Stage 1-2 ========")
    with open(log_path, "a") as file:
        file.write(f"======== Stage 1-1 ========\n")
    for epoch in range(st2_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for X_img, Y_img, _, _ in train_loader:
            if(debug and n_batches >= 1):
                break
            # bilinear upscale X_img to Y_img size
            X_img = F.interpolate(X_img, size=(H_h, W_h), mode='bilinear', align_corners=False).to(device).float()
            X_img = torch.clamp(X_img, 0.0, 1.0)
            pred = model(X_img, stage=1)
            if not torch.isfinite(pred).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            Y_img = Y_img.to(device).float()
            loss = criterion(pred, Y_img)
            if not torch.isfinite(loss).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            loss /= microbatch_steps
            epoch_loss += loss.item()
            batch_loss += loss.item()
            scaler.scale(loss).backward()
            if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                scheduler.step()
            n_batches += 1
            if(n_batches % 500 == 0):
                with open(log_path, "a") as file:
                    file.write(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}\n")
                print(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}")
                batch_loss = 0.0
        print(f"Stage 1-2 Epoch {epoch + 1}/{st2_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}")
        with open(log_path, "a") as file:
            file.write(f"Stage 1-2 Epoch {epoch + 1}/{st2_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}\n")
    del optimizer
    del scheduler
    # validation
    if val_loader is not None and debug == False:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for X_img, Y_img, _, _ in val_loader:
            # bilinear upscale X_img to Y_img size
                X_img = F.interpolate(X_img, size=(H_h, W_h), mode='bilinear', align_corners=False).to(device).float()
                X_img = torch.clamp(X_img, 0.0, 1.0)
                pred = model(X_img, stage=1)
                Y_img = Y_img.to(device).float()
                loss = criterion(pred, Y_img)
                val_loss += loss.item()
                n_val += 1
            print(f"Stage 1 Validation Loss: {val_loss / n_val:.4f}")
            with open(log_path, "a") as file:
                file.write(f"Stage 1 Validation Loss: {val_loss / n_val:.4f}\n")
    savepoint(model, stage=1, epoch=st1_epochs + st2_epochs)
    return model

def stage_2_train(model, train_loader, val_loader = None, lr = 1e-4, total_epochs = 10, microbatch_steps = 4):
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)
    strikes = 0
    # train learner against teacher, L2
    # only model.learner trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.learner.parameters():
        param.requires_grad = True
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.learner.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type, enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    model.to(device)
    model.train()

    print("======== Stage 2 ========")
    with open(log_path, "a") as file:
        file.write(f"======== Stage 2 ========\n")
    best_val_loss = float("inf")
    for epoch in range(total_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for _, Y_img, _, _ in train_loader:
            if(debug and n_batches >= 1):
                break
            Y_img = Y_img.to(device).float()
            pred_dict = model(Y_img, stage=2) #{"x_prep": x_prep, "x1": x1, "x2": x2, "lea224": lea224, "lea112": lea112, "lea56": lea56}
            # match pred_dict["lea224"] against pred_dict["x_prep"], ["lea112"] against ["x1"], ["lea56"] against ["x2"]
            if not torch.isfinite(pred_dict["lea224"]).all() or not torch.isfinite(pred_dict["lea112"]).all() or not torch.isfinite(pred_dict["lea56"]).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            loss = criterion(pred_dict["lea224"], pred_dict["x_prep"]) + criterion(pred_dict["lea112"], pred_dict["x1"]) + criterion(pred_dict["lea56"], pred_dict["x2"])
            if not torch.isfinite(loss).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            loss /= microbatch_steps
            epoch_loss += loss.item()
            batch_loss += loss.item()
            scaler.scale(loss).backward()
            if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
            n_batches += 1
            if(n_batches % 500 == 0):
                with open(log_path, "a") as file:
                    file.write(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}\n")
                print(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}")
                batch_loss = 0.0
        print(f"Stage 2 Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}")
        with open(log_path, "a") as file:
            file.write(f"Stage 2 Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}\n")

        # validation
        if val_loader is not None and debug == False:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val = 0
                for _, Y_img, _, _ in val_loader:
                    Y_img = Y_img.to(device).float()
                    pred_dict = model(Y_img, stage=2)
                    loss = criterion(pred_dict["lea224"], pred_dict["x_prep"]) + criterion(pred_dict["lea112"], pred_dict["x1"]) + criterion(pred_dict["lea56"], pred_dict["x2"])
                    val_loss += loss.item()
                    n_val += 1
                print(f"Stage 2 epoch {epoch} Validation Loss: {val_loss / n_val:.4f}")
                with open(log_path, "a") as file:
                    file.write(f"Stage 2 epoch {epoch} Validation Loss: {val_loss / n_val:.4f}\n")
                if val_loss / n_val < best_val_loss:
                    best_val_loss = val_loss / n_val
                    savepoint(model, stage=2, epoch=epoch + 1)
    del optimizer
    del scheduler
    return model

def stage_3_train(model, train_loader, val_loader = None, lr = 1e-4, total_epochs = 5, microbatch_steps = 4, use_percep = True, crit_perc = 0.1,
                  epochs_per_stage = (2, 2, 1)):
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)
    strikes = 0
    # train decoder against teacher, VGG + slight L1 loss
    # model.teacher, model.teacher_upscaler, model.learner not trainable
    assert sum(epochs_per_stage) == total_epochs, "Sum of epochs_per_stage must equal total_epochs"
    
    exclusions = ["teacher", "teacher_upscaler", "learner", "enc4", "enc3", "enc2"]
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if any(excl in name for excl in exclusions):
            param.requires_grad = False
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    
    stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"])
    
    criterion = nn.L1Loss()
    if use_percep:
        perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
        perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
        perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
        perceptual_criterion.eval()
    scaler = torch.amp.GradScaler(device_type, enabled=True)
    model.to(device)
    model.train()

    for stage_idx in range(len(epochs_per_stage)):
        print(f"======== Stage 3-{stage_idx + 1} ========")
        with open(log_path, "a") as file:
            file.write(f"======== Stage 3-{stage_idx + 1} ========\n")
            file.write(f"Unfreezing layers: {stages[stage_idx]}\n")
        for layer_name in stages[stage_idx]:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(list(trainable_params), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
        best_val_loss = float("inf")
        for epoch in range(epochs_per_stage[stage_idx]):
            # unfreeze according to stage
            optimizer.zero_grad(set_to_none=True)
            n_batches = 0
            epoch_loss = 0.0
            batch_loss = 0.0
            for _, Y_img, _, _ in train_loader:
                if(debug and n_batches >= 1):
                    break
                Y_img = Y_img.to(device).float()
                # staged training; decoder always trainable, enc2~enc4 progressively unfrozen
                pred = model(Y_img, stage=3)
                if not torch.isfinite(pred).all():
                    print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                    with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                        file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                    optimizer.zero_grad(set_to_none=True)
                    n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                    # 3 strikes
                    if(strikes >= 3):
                        raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                    strikes += 1
                    continue
                loss = criterion(pred, Y_img)
                if not torch.isfinite(loss).all():
                    print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                    with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                        file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                    optimizer.zero_grad(set_to_none=True)
                    n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                    # 3 strikes
                    if(strikes >= 3):
                        raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                    strikes += 1
                    continue
                if use_percep:
                    perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                    perceptual_criterion.eval()
                    loss_perc = perceptual_criterion(pred, Y_img)
                    loss = loss_perc + crit_perc * loss
                loss /= microbatch_steps
                epoch_loss += loss.item()
                batch_loss += loss.item()
                scaler.scale(loss).backward()
                if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    scheduler.step()
                n_batches += 1
                if(n_batches % 500 == 0):
                    with open(log_path, "a") as file:
                        file.write(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}\n")
                    print(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}")
                    batch_loss = 0.0
            print(f"Stage 3-{stage_idx + 1} Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}")
            with open(log_path, "a") as file:
                file.write(f"Stage 3-{stage_idx + 1} Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}\n")
            # validation
            if val_loader is not None and debug == False:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    n_val = 0
                    for _, Y_img, _, _ in val_loader:
                        Y_img = Y_img.to(device).float()
                        pred = model(Y_img, stage=3)
                        loss = criterion(pred, Y_img)
                        if use_percep:
                            perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                            perceptual_criterion.eval()
                            loss_perc = perceptual_criterion(pred, Y_img)
                            loss = loss_perc + crit_perc * loss
                        val_loss += loss.item()
                        n_val += 1
                    print(f"Stage 3-{stage_idx + 1} Epoch {epoch + 1}/{total_epochs}, Validation Loss: {val_loss / n_val:.4f}")
                    with open(log_path, "a") as file:
                        file.write(f"Stage 3-{stage_idx + 1} Epoch {epoch + 1}/{total_epochs}, Validation Loss: {val_loss / n_val:.4f}\n")
                    if val_loss / n_val < best_val_loss:
                        best_val_loss = val_loss / n_val
                        savepoint(model, stage=3, epoch=sum(epochs_per_stage[:stage_idx]) + epoch + 1)
        del optimizer
        del scheduler
    return model

def stage_4_train(model, train_loader, val_loader = None, lr = 1e-5, total_epochs = 3, microbatch_steps = 4, use_percep = True, perc = 0.1):
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(False)
    strikes = 0
    # fine tune decoder against learner, VGG + slight L1 loss
    # model.teacher, model.teacher_upscaler not trainable
    exclusions = ["teacher", "teacher_upscaler"]
    for param in model.parameters():
        param.requires_grad = True
        for name, param in model.named_parameters():
            if any(excl in name for excl in exclusions):
                param.requires_grad = False
    trainable_params = [param for param in model.parameters() if param.requires_grad]
        
    criterion = nn.L1Loss()
    if use_percep:
        perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
        perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
        lambda_perc = perc
        perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
        perceptual_criterion.eval()
    else:
        perceptual_criterion = None
        lambda_perc = 0.0
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type, enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    model.to(device)
    model.train()

    print("======== Stage 4 ========")
    with open(log_path, "a") as file:
        file.write(f"======== Stage 4 ========\n")
    best_val_loss = float("inf")
    for epoch in range(total_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for _, Y_img, _, _ in train_loader:
            if(debug and n_batches >= 1):
                break
            Y_img = Y_img.to(device).float()
            pred = model(Y_img, stage=4)
            if not torch.isfinite(pred).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite logits; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            loss = criterion(pred, Y_img)
            if not torch.isfinite(loss).all():
                print(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping")
                with open(f"logs/training/{desc_path}{desc}.txt", "a") as file:
                    file.write(f"----WARNING: [Batch {n_batches}] Returned infinite loss; skipping\n")
                optimizer.zero_grad(set_to_none=True)
                n_batches -= (n_batches% microbatch_steps)  # reset microbatch count
                # 3 strikes
                if(strikes >= 3):
                    raise RuntimeError("Exceeded maximum number of infinite loss strikes; aborting training")
                strikes += 1
                continue
            if use_percep:
                perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                perceptual_criterion.eval()
                loss_perc = perceptual_criterion(pred, Y_img)
                loss += lambda_perc * loss_perc
            loss /= microbatch_steps
            epoch_loss += loss.item()
            batch_loss += loss.item()
            scaler.scale(loss).backward()
            if((n_batches + 1) % microbatch_steps == 0 or (n_batches + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                scheduler.step()
            n_batches += 1
            if(n_batches % 500 == 0):
                with open(log_path, "a") as file:
                    file.write(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}\n")
                print(f"    Batch {n_batches}, Training Loss: {batch_loss * microbatch_steps / 500:.4f}")
                batch_loss = 0.0
        print(f"Stage 4 Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}")
        with open(log_path, "a") as file:
            file.write(f"Stage 4 Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss * microbatch_steps / n_batches:.4f}\n")
        # validation
        if val_loader is not None and debug == False:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val = 0
                for _, Y_img, _, _ in val_loader:
                    Y_img = Y_img.to(device).float()
                    pred = model(Y_img, stage=4)
                    loss = criterion(pred, Y_img)
                    if use_percep:
                        perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                        perceptual_criterion.eval()
                        loss_perc = perceptual_criterion(pred, Y_img)
                        loss += lambda_perc * loss_perc
                    val_loss += loss.item()
                    n_val += 1
                print(f"Stage 4 Validation Loss: {val_loss / n_val:.4f}")
                with open(log_path, "a") as file:
                    file.write(f"Stage 4 Validation Loss: {val_loss / n_val:.4f}\n")
                if val_loss / n_val < best_val_loss:
                    best_val_loss = val_loss / n_val
                    savepoint(model, stage=4, epoch=epoch + 1)
    del optimizer
    del scheduler
    return model

if __name__ == "__main__":
    if not os.path.exists(f"./logs/training/{desc_path}"):
        os.makedirs(f"./logs/training/{desc_path}", exist_ok=True)
    # prepare data loaders
    train_dataset = FairFaceDataset(train_image_path, train_label_path, lr_size = (sz, sz))
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = FairFaceDataset(val_image_path, val_label_path, lr_size = (sz, sz))
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, num_workers=8, pin_memory=True)
    # check device
    print(f"Using device: {device_type}")
    # stage-wise training
    print("Starting training")
    with open(log_path, "a") as file:
        file.write(f"Training started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
        file.write(f"Configuration: {model_type}  {config_str}\n")
    if debug:
        print("DEBUG RUN")
    model = stage_1_train(model, train_loader, val_loader, st1_epochs=3, st2_epochs=2, microbatch_steps=microbatches)
    model = stage_2_train(model, train_loader, val_loader, total_epochs=epoch_stages[1], microbatch_steps=microbatches)
    model = stage_3_train(model, train_loader, val_loader, use_percep=use_percep, crit_perc=perc, total_epochs=epoch_stages[2], microbatch_steps=microbatches, epochs_per_stage=decoder_stages)
    model = stage_4_train(model, train_loader, val_loader, use_percep=use_percep, perc=perc, total_epochs=epoch_stages[3], microbatch_steps=microbatches)
    print("Training complete.")
