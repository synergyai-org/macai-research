import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################
### Loss Functions ###############################################
##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2.0, class_weights=None, reduction='mean'):
        """
        smoothing: float, 0~1. 라벨 스무딩 계수
        gamma: float, >=0. Focal Loss의 focusing parameter.
        class_weights: Tensor of shape [num_classes], e.g. [1.0, 2.0, 0.5, ...]
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.class_weights = class_weights  # [C]
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B, C] with -1 where labels are missing
        """
        # === 유효 라벨 마스크 생성 ===
        mask = (targets != -1)  # [B, C]

        # === 라벨 스무딩 적용 ===
        smoothed_targets = torch.where(
            mask,
            targets * (1 - self.smoothing) + self.smoothing * 0.5, # Two-sided smoothing
            torch.zeros_like(targets)  # dummy, masked out later anyway
        )

        # === BCE Loss 계산 (리덕션 없이) ===
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction='none'
        )  # [B, C]

        # === Focal Loss 로직 추가 ===
        # p_t는 정답 레이블에 대한 예측 확률입니다.
        # p_t = exp(-bce_loss) 근사를 통해 계산할 수 있습니다.
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t).pow(self.gamma) * bce_loss

        # === 마스킹 적용 ===
        focal_loss = focal_loss * mask  # [B, C]

        # === 클래스 가중치 적용 ===
        if self.class_weights is not None:
            # class_weights를 올바른 디바이스로 이동
            device = focal_loss.device
            class_weights = self.class_weights.to(device)
            focal_loss = focal_loss * class_weights.view(1, -1)  # [1, C] broadcast

        # === 리덕션 ===
        if self.reduction == 'mean':
            # 유효한 요소들의 개수로만 나누어 평균을 계산합니다.
            return focal_loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # [B, C]
        

import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxFocalSmoothBCELoss(nn.Module):
    def __init__(self, 
                 pos_weight=None,
                 task_weights=None,
                 gamma=2.0,
                 smoothing=0.0,
                 reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.task_weights = task_weights
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B, C] with -1 where labels are missing
        """
        # 디바이스 일치시키기 (is not None 으로 수정)
        if self.pos_weight is not None and self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
        
        if self.task_weights is not None and self.task_weights.device != inputs.device:
            self.task_weights = self.task_weights.to(inputs.device)

        # 마스크를 먼저 정의
        mask = (targets != -1)

        # 라벨 스무딩을 위한 타겟 복사 및 수정 (가독성/안전성)
        bce_targets = targets.clone()
        if self.smoothing > 0:
            # 유효한 레이블에만 스무딩 적용
            bce_targets[mask] = bce_targets[mask] * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 1. BCE Loss 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, bce_targets,  # 스무딩된 복사본 사용
            pos_weight=self.pos_weight, 
            reduction='none'
        )

        # 2. Focal Loss 적용
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t).pow(self.gamma) * bce_loss

        # 3. 태스크 가중치 적용 (else 구문 추가하여 NameError 해결)
        if self.task_weights is not None:
            weighted_loss = focal_loss * self.task_weights.view(1, -1)
        else:
            weighted_loss = focal_loss

        # 4. 마스킹 및 최종 리덕션
        final_loss = weighted_loss * mask
        
        if self.reduction == 'mean':
            if mask.sum() > 0:
                return final_loss.sum() / mask.sum()
            else:
                return torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return final_loss.sum()
        else:
            return final_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class AuxFocalBCELoss(nn.Module):
    def __init__(self, 
                 pos_weight=None,    # [num_classes] 형태의 미리 계산된 텐서
                 task_weights=None,
                 gamma=2.0,       # Focal Loss 파라미터
                 reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.task_weights = task_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B, C] with -1 where labels are missing
        """
        # 디바이스 일치시키기
        if self.pos_weight:
            if self.pos_weight.device != inputs.device:
                self.pos_weight = self.pos_weight.to(inputs.device)

        # 1. pos_weight를 적용하여 BCE Loss 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight, 
            reduction='none'
        )

        # 2. Focal Loss 적용
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t).pow(self.gamma) * bce_loss

        # 3. 태스크 가중치 적용
        num_classes = inputs.shape[1]
        if self.task_weights is not None:
            weighted_loss = focal_loss * self.task_weights.view(1, -1)

        # 4. 마스킹 및 최종 리덕션
        mask = (targets != -1)
        final_loss = weighted_loss * mask
        
        if self.reduction == 'mean':
            # 유효한 요소들의 개수로만 나누어 평균 계산
            return final_loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return final_loss.sum()
        else:
            return final_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. BCE loss (logits 기반이므로 AMP-safe)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # [B, C] 형태

        # 2. pt = exp(-bce) 방식으로 안정적인 확률값 계산 (pt ∈ (0, 1])
        pt = torch.exp(-bce_loss)

        # 3. Focal weight 계산 (수치적으로 안정적)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)

        # 4. Focal loss
        loss = focal_weight * bce_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: Tensor of shape (num_classes,) or float — class별 weight
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (B, C), raw logits
        targets: (B, C), binary labels (0 or 1)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # (B, C)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)  # (B, C)

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = torch.where(targets == 1, alpha, 1 - alpha)  # (B, C)
            focal_loss = at * (1 - pt) ** self.gamma * BCE_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None, reduction='mean'):
        """
        smoothing: float, 0~1
        class_weights: Tensor of shape [num_classes], e.g. [1.0, 2.0, 0.5, ...]
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights  # [C]
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B, C] with -1 where labels are missing
        """
        # === 유효 라벨 마스크 생성 ===
        mask = (targets != -1)  # [B, C]
        # print(mask.shape)
        # === 라벨 스무딩 적용 ===
        smoothed_targets = torch.where(
            mask,
            targets * (1 - self.smoothing) + self.smoothing,
            torch.zeros_like(targets)  # dummy, masked out later anyway
        )
        # print(smoothed_targets.shape)   
        # === BCE Loss 계산 ===
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction='none'
        )  # [B, C]
        # print(bce_loss.shape)   

        # === 마스킹 적용 ===
        bce_loss = bce_loss * mask  # [B, C]
        # print(bce_loss.shape)   

        # === 클래스 가중치 적용 ===
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights.view(1, -1)  # [1, C] broadcast
        # print(bce_loss.shape)   

        # === 리덕션 ===
        if self.reduction == 'mean':
            return bce_loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss  # [B, C]




class DynamicWeightedBCELoss(nn.Module):
    def __init__(self):
        super(DynamicWeightedBCELoss, self).__init__()

    def forward(self, inputs, targets):
        pos_weight = targets.mean()  # Positive 비율
        neg_weight = 1 - pos_weight  # Negative 비율
        weights = pos_weight * targets + neg_weight * (1 - targets)
        BCE_loss = nn.BCEWithLogitsLoss(weight=weights)(inputs, targets)
        return BCE_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

class SymmetricFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(SymmetricFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        probs = torch.sigmoid(inputs)
        focal_term = (1 - probs) ** self.gamma * targets + probs ** self.gamma * (1 - targets)
        return (focal_term * BCE_loss).mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D]
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss
