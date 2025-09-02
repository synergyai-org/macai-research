from models import SynAI
import torch

def model_build(model_info):
    model = SynAI.MAE_1D_250409_v3(
        seq_length  = model_info['seq_length'],
        in_channels = model_info['in_channels'],
        patch_size  = model_info['patch_size'],
        embed_dim   = model_info['embed_dim'],
        merge_mode  = model_info['merge_mode'],  # linear_projection avg add
        encoder     = model_info['encoder'],
    )
    model = SynAI.OnlyEncoderForFT_250409(
        model,
        num_classes = model_info['num_classes'],
        embed_dim = model_info['embed_dim'],
    )
    return model

def load_models(model_path_list, model_info, device='cpu'):
    models = []
    for model_path in model_path_list:
        checkpoint = torch.load(model_path, weights_only=True)
        model = model_build(model_info).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append({'model_path': model_path, 'model': model})
    return models


from torch.utils.data import DataLoader
from processing import transforms, dataset

transform_pipeline = {}
transform_pipeline['val'] = transforms.TransformPipeline([
    transforms.NormalizeECG(method="tanh", scope="lead-wise", scale=1),
])


import torch
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def inference_tensor(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    activation: Optional[str] = 'sigmoid',
    to_cpu: bool = True
) -> Dict[str, torch.Tensor]:
    """
    단일 샘플 또는 배치 텐서에 대해 추론 결과 반환
    """
    original_input = input_tensor.clone().detach()
    device = next(model.parameters()).device
    model.eval()
    inference_input = input_tensor.to(device)
    is_batch = inference_input.dim() >= 3
    if not is_batch:
        inference_input = inference_input.unsqueeze(0)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        logits = model(inference_input)
    if not is_batch:
        logits = logits.squeeze(0)
    probs = None
    if activation == 'sigmoid':
        probs = torch.sigmoid(logits)
    elif activation == 'softmax':
        probs = torch.softmax(logits, dim=-1)
    if to_cpu:
        logits = logits.cpu()
        if probs is not None:
            probs = probs.cpu()
    result = {
        "input": original_input,
        "logits": logits
    }
    if probs is not None:
        result["probs"] = probs
    return result


import numpy as np

def classify_risk(
    probs:np.array, high_thr:float, inter_thr:float, method:str=None):
    '''
        probs: 위험도 확률 리스트
        high_thr: 고위험군 기준 임계값 (상위 10%)
        inter_thr: 중간위험군 기준 임계값 (상위 10% ~ 상위 30%)
    '''
    results = []    
    if method == 'mean':
        probs = probs.mean()
        if probs >= high_thr:
            return [probs, "High"]
        elif probs >= inter_thr:
            return [probs, "Intermediate"]
        else:
            return [probs, "Low"]
    else:
        for prob in probs:
            if prob >= high_thr:
                results.append([prob, "High"])
            elif prob >= inter_thr:
                results.append([prob, "Intermediate"])
            else:
                results.append([prob, "Low"])
        return results