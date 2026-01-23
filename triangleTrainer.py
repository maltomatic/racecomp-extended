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

from loaders.load_redux import FairFaceDataset_Trim as FairFaceDataset
from loaders.load_redux import classes, class_count, train_image_path, train_label_path, val_image_path, val_label_path

from toolkit.VGGPerceptionLoss import PerceptualLossVGG19
from toolkit.debugs import denormalize_imagenet
from toolkit.criteria import psnr, ssim_simple

from models.triangleNet import TriangleNet

B = 32
C = 3
H_l = W_l = 56
H_h = W_h = 224

#################### configs #################### 
TRAINING = True
debug = True
resume = False
training_comment = "Testing TriangleNet"

# train_list = ["All", "East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
train_list = ["All"]
use_percep = True
perc = 0.1
use_ssim = False
microbatches = 4 # 1 for no microbatching, n for n-step microbatching, max 8 recommended to avoid gradient explosion
sz = 56 # or 56
epoch_stages = (5, 10, 5, 3)
# stage 1: fine-tune teacher, L1 loss
# stage 2: train learner against teacher, L2 + slight VGG loss
# stage 3: train decoder against teacher, VGG + slight L1 loss
# stage 4: fine tune decoder against learner, VGG + slight L1 loss
decoder_stages = (2, 2, 1, 0) # encoder unfreeze level when training decoder in s3

under_represented_ratio = 0.05
tgt_race = "All"
test_stage = 2
test_epoch = 4

config_str = f"size{sz}_mb{microbatches}_percep{int(use_percep)}_ssim{int(use_ssim)}"
#################################################

model = TriangleNet(px_shuffle=False, px_shuffle_interpolate=True, training = True)
model_type = "TriangleNet"
desc_path = f"{model_type}/{config_str}/"
desc = f"trained_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.autograd.set_detect_anomaly(False)

def imagenet_denorm(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1,-1,1,1)
    std  = x.new_tensor(IMAGENET_STD).view(1,-1,1,1)
    return x * std + mean

def accumulate_by_race(bucket, race, loss, psnr_val, ssim_val):
    b = bucket.setdefault(race, {"loss": [], "psnr": [], "ssim": []})
    b["loss"].append(loss); b["psnr"].append(psnr_val); b["ssim"].append(ssim_val)

def stage_1_train(model, train_loader, val_loader = None, st1_lr = 3e-4, st2_lr = 1e-5, st1_epochs = 7, st2_epochs = 3, microbatch_steps = 4):
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
    for epoch in range(st1_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for _, Y_img, _, _ in train_loader:
            Y_img = Y_img.to(device).float()
            pred = model(Y_img, stage=1)
            loss = criterion(pred, Y_img)
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
            if(n_batches % 100 == 0):
                print(f"    Batch {n_batches}, Training Loss: {batch_loss / 100:.4f Loss}")
                batch_loss = 0.0
        print(f"Stage 1-1 Epoch {epoch + 1}/{st1_epochs}, Training Loss: {epoch_loss / n_batches:.4f Loss}")
    del optimizer
    del scheduler
    
    for param in model.teacher.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(list(model.teacher.parameters()) + list(model.teacher_upscaler.parameters()), lr=st2_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, st2_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    for epoch in range(st2_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for _, Y_img, _, _ in train_loader:
            Y_img = Y_img.to(device).float()
            pred = model(Y_img, stage=1)
            loss = criterion(pred, Y_img)
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
            if(n_batches % 100 == 0):
                print(f"    Batch {n_batches}, Training Loss: {batch_loss / 100:.4f Loss}")
                batch_loss = 0.0
        print(f"Stage 1-2 Epoch {epoch + 1}/{st2_epochs}, Training Loss: {epoch_loss / n_batches:.4f Loss}")
    del optimizer
    del scheduler
    # validation
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for _, Y_img, _, _ in val_loader:
                Y_img = Y_img.to(device).float()
                pred = model(Y_img, stage=1)
                loss = criterion(pred, Y_img)
                val_loss += loss.item()
                n_val += 1
            print(f"Stage 1 Validation Loss: {val_loss / n_val:.4f}")

def stage_2_train(model, train_loader, val_loader = None, lr = 1e-4, total_epochs = 10, microbatch_steps = 4, use_percep = True, perc = 0.1):
    # train learner against teacher, L2 + slight VGG loss
    # only model.learner trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.learner.parameters():
        param.requires_grad = True
    
    criterion = nn.MSELoss()
    if use_percep:
        perc_layers = (4, 8, 12, 16)  # conv2_2, conv3_4, conv4_4, conv5_4
        perc_weights = {4: 1.0, 8: 1.0, 12: 1.0, 16: 1.0}  # equal weighting
        lambda_perc = perc
        perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
        perceptual_criterion.eval()
    else:
        perceptual_criterion = None
        lambda_perc = 0.0
    optimizer = torch.optim.AdamW(model.learner.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type, enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs * math.ceil(len(train_loader) / microbatch_steps)))
    model.to(device)
    model.train()

    for epoch in range(total_epochs):
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0
        epoch_loss = 0.0
        batch_loss = 0.0
        for _, Y_img, _, _ in train_loader:
            Y_img = Y_img.to(device).float()
            pred_dict = model(Y_img, stage=2) #{"x_prep": x_prep, "x1": x1, "x2": x2, "lea224": lea224, "lea112": lea112, "lea56": lea56}
            # match pred_dict["lea224"] against pred_dict["x_prep"], ["lea112"] against ["x1"], ["lea56"] against ["x2"]
            loss = criterion(pred_dict["lea224"], pred_dict["x_prep"]) + criterion(pred_dict["lea112"], pred_dict["x1"]) + criterion(pred_dict["lea56"], pred_dict["x2"])
            if use_percep:
                perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                perceptual_criterion.eval()
                loss_perc = perceptual_criterion(pred_dict["lea224"], pred_dict["x_prep"]) + perceptual_criterion(pred_dict["lea112"], pred_dict["x1"]) + perceptual_criterion(pred_dict["lea56"], pred_dict["x2"])
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
            if(n_batches % 100 == 0):
                print(f"    Batch {n_batches}, Training Loss: {batch_loss / 100:.4f Loss}")
                batch_loss = 0.0
        print(f"Stage 2 Epoch {epoch + 1}/{total_epochs}, Training Loss: {epoch_loss / n_batches:.4f Loss}")
    del optimizer
    del scheduler
    # validation
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for _, Y_img, _, _ in val_loader:
                Y_img = Y_img.to(device).float()
                pred_dict = model(Y_img, stage=2)
                loss = criterion(pred_dict["lea224"], pred_dict["x_prep"]) + criterion(pred_dict["lea112"], pred_dict["x1"]) + criterion(pred_dict["lea56"], pred_dict["x2"])
                if use_percep:
                    perceptual_criterion = PerceptualLossVGG19(layer_weights=perc_weights, layers=perc_layers).to(device)
                    perceptual_criterion.eval()
                    loss_perc = perceptual_criterion(pred_dict["lea224"], pred_dict["x_prep"]) + perceptual_criterion(pred_dict["lea112"], pred_dict["x1"]) + perceptual_criterion(pred_dict["lea56"], pred_dict["x2"])
                    loss += lambda_perc * loss_perc
                val_loss += loss.item()
                n_val += 1
            print(f"Stage 2 Validation Loss: {val_loss / n_val:.4f}")

def stage_3_train(model, train_loader, val_loader = None, lr = 1e-4, total_epochs = 5, microbatch_steps = 4, use_percep = True, perc = 0.1,
                  epochs_per_stage = (2, 2, 1)):
    # train decoder against teacher, VGG + slight L1 loss
    # model.teacher, model.teacher_upscaler, model.learner not trainable
    assert sum(epochs_per_stage) == total_epochs, "Sum of epochs_per_stage must equal total_epochs"
    
    trainable_params = []
    for param in model.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in model.teacher.parameters():
        param.requires_grad = False
        trainable_params.remove(param)
    for param in model.teacher_upscaler.parameters():
        param.requires_grad = False
        trainable_params.remove(param)
    for param in model.learner.parameters():
        param.requires_grad = False
        trainable_params.remove(param)
    
    stages=(["enc4"], ["enc4","enc3"], ["enc4","enc3","enc2"])
    
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

    for epoch in range(total_epochs):
        for _, Y_img, _, _ in train_loader:
            Y_img = Y_img.to(device).float()
            # staged training; decoder always trainable, encoder unfreeze over stages

def stage_4_train(model, train_loader, val_loader = None, lr = 1e-5, total_epochs = 3, microbatch_steps = 4, use_percep = True, perc = 0.1):
    # fine tune decoder against learner, VGG + slight L1 loss
    # model.teacher, model.teacher_upscaler not trainable
    trainable_params = []
    for param in model.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in model.teacher.parameters():
        param.requires_grad = False
        trainable_params.remove(param)
    for param in model.teacher_upscaler.parameters():
        param.requires_grad = False
        trainable_params.remove(param)
        
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

    for epoch in range(total_epochs):
        ... #TODO