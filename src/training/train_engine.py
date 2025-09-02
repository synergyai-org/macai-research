##################################################################
### Training Functions ###########################################
##################################################################

from typing import Callable, Dict
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.amp as amp
import numpy as np

def train_one_epoch(
        model, 
        data_loader, 
        loss_fn, 
        device, 
        amp_scaler, 
        optimizer,
        clip_grad_norm = 1.0,
        scheduler_lr = None,
        verbose = True,  # verbose 옵션 추가
    ):
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not verbose)  # verbose가 False면 tqdm 비활성화
    status = {'loss':0, 'lr':None, 'sch_lr':[],}
    all_logits, all_labels = [], []

    model.train(True)
    for _, batch in pbar:
        inputs, labels = batch['input'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            logits = model(inputs)            
            loss = loss_fn(logits, labels)            

        # loss.backward -> opt.unscale_-> opt.step -> opt.zero_grad 
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        status['loss'] += loss.item()
        monitor = f"| TRN | batch loss : {loss.item():.4f}, "
        if scheduler_lr:
            scheduler_lr.step()
            status['sch_lr'].append(scheduler_lr.get_last_lr()[0])
            monitor += f"lr : {status['sch_lr'][-1]:.1e} "
        else:
            current_lr = optimizer.param_groups[0]['lr']
            status['sch_lr'].append(current_lr)
        pbar.set_description(monitor)    

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if scheduler_lr: 
        status['lr'] = np.mean(status['sch_lr']).item()
    status['loss'] = status['loss'] / len(data_loader)
    status['logits'] = torch.cat(all_logits, dim=0)
    status['labels'] = torch.cat(all_labels, dim=0)

    return status


@torch.no_grad()
def evaluate_one_epoch(
        model, 
        data_loader, 
        loss_fn, 
        device, 
        verbose = True,  # verbose 옵션 추가
    ):
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not verbose)  # verbose가 False면 tqdm 비활성화
    status = {'loss':0}
    all_logits, all_labels = [], []

    model.eval()
    for _, batch in pbar:
        inputs, labels = batch['input'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type):
            logits = model(inputs)            
            loss = loss_fn(logits, labels)            

        status['loss'] += loss.item()
        monitor = f"| VAL | batch loss: {loss.item():.4f}, "
        pbar.set_description(monitor)    

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    status['loss'] = status['loss'] / len(data_loader)
    status['logits'] = torch.cat(all_logits, dim=0)
    status['labels'] = torch.cat(all_labels, dim=0)

    return status




@torch.no_grad()
def only_inference(
        model, data_loader, device, 
    ):
    all_logits, all_labels = [], []
    model.eval()
    for current_step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs, labels = batch['input'], batch['label']
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type):
            logits = model(inputs)

        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)  
    return all_logits, all_labels
