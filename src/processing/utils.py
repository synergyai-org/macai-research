import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import cycle


def SetSeedEverything(seed, fully_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False    



#---------------###################################################
#--- Metrics ---###################################################
#---------------###################################################

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score,
    precision_recall_curve, roc_curve, average_precision_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import pandas as pd


def to_numpy(x):
    """Tensor → Numpy 변환 + SafeType (float16 → float32)"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    # float 타입 안정성 확보
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    return x

def normalize_probs(y_pred):
    """Softmax numeric safety normalization"""
    row_sum = np.sum(y_pred, axis=1, keepdims=True)
    return y_pred / row_sum

def find_optimal_cutoff(y_true, y_pred, method='f1'):
    """각 클래스별 optimal threshold 계산"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.5  # 전부 동일 label일 경우 기본값 반환

    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        thresholds = np.append(thresholds, 1.0)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.nanargmax(f1_scores)
        return thresholds[optimal_idx]

    elif method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        return thresholds[optimal_idx]

    elif method == 'mcc':
        thresholds = np.linspace(0, 1, 101)
        mccs = []
        for th in thresholds:
            pred_binary = (y_pred >= th).astype(int)
            mccs.append(matthews_corrcoef(y_true, pred_binary))
        optimal_idx = np.nanargmax(mccs)
        return thresholds[optimal_idx]

    elif method == 'distance':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_idx = np.argmin(distances)
        return thresholds[optimal_idx]

    else:
        raise ValueError("Invalid method. Choose from ['f1', 'youden', 'mcc', 'distance']")
    

def classification_metrics(
    y               : torch.Tensor, 
    y_pred          : torch.Tensor, 
    activation_fn   : bool = True, 
    mode            : str = 'multilabel',  # 'multilabel', 'multiclass'
    threshold_method: str = 'f1'  # 'f1', 'youden', 'mcc', 'distance'
):

    if activation_fn:
        if mode == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
        elif mode == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=-1)
    y, y_pred = to_numpy(y), to_numpy(y_pred)

    results = {}

    cls_count = y_pred.shape[-1]
    for i in range(cls_count):
        _y, _y_pred = y[:, i], y_pred[:, i]
        mask = (_y != -1)
        _y, _y_pred = _y[mask], _y_pred[mask]
        if len(_y) == 0:
            continue  # 유효한 샘플 없으면 skip
        
        # metric 계산
        try:
            auroc = roc_auc_score(_y, _y_pred)
        except ValueError:
            auroc = np.nan
        try:
            auprc = average_precision_score(_y, _y_pred)
        except ValueError:
            auprc = np.nan

        # optimal cutoff 찾기
        best_th = find_optimal_cutoff(_y, _y_pred, method=threshold_method)
        _y_pred_binary = (_y_pred >= best_th).astype(int)

        accuracy = accuracy_score(_y, _y_pred_binary)
        precision_val = precision_score(_y, _y_pred_binary, zero_division=0)
        recall_val = recall_score(_y, _y_pred_binary, zero_division=0)
        f1_val = f1_score(_y, _y_pred_binary, zero_division=0)

        cm = confusion_matrix(_y, _y_pred_binary, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # 결과 기록
        results[i] = {
            'AUROC'     : auroc,
            'AUPRC'     : auprc,
            'opt_thr'   : best_th, 
            'Accu'      : accuracy,
            'Sens'      : recall_val,
            'Prec'      : precision_val,
            'Spec'      : specificity,
            'NPV'       : npv,
            'F1'        : f1_val,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Total_N'   : tn+fp+fn+tp
        }

    # === Macro-average ===
    metrics = ['AUROC','AUPRC','Accu','Sens','Prec','Spec','NPV','F1']
    results['avg'] = {}
    for m in metrics:
        vals = [results[k].get(m, np.nan) 
                for k in results.keys() if str(k).isdigit()]
        vals = [v for v in vals if not np.isnan(v)]
        results['avg'][m] = np.mean(vals) if vals else np.nan

    return results


def history_recording(
    record_df: pd.DataFrame,
    epoch: int,
    train_metrics: dict,
    valid_metrics: dict,
    save_path: str,
    save_each_epoch: bool = True,
    scheduler: bool = False,
    wd_scheduler: bool = False,
    ema_scheduler: bool = False,
) -> pd.DataFrame:
    """
    Record training/validation metrics for each epoch.
    """

    # === 1) 새 행(dict) 구성 ===
    row_data = {"epoch": epoch,
                "train_loss": train_metrics.get("loss", None),
                "valid_loss": valid_metrics.get("loss", None)}

    for prefix, metrics_dict in [("train", train_metrics), ("valid", valid_metrics)]:
        for cls_num, cls_metrics in metrics_dict.items():
            if cls_num == "loss":
                continue
            for metric_name, v in cls_metrics.items():
                col_name = f"{prefix}_{metric_name}_{cls_num}"
                row_data[col_name] = v

    # === 2) dict → DataFrame 변환 ===
    new_row = pd.DataFrame([row_data])

    # === 3) concat으로 추가 ===
    record_df = pd.concat([record_df, new_row], ignore_index=True)

    # === 4) 저장 ===
    if save_each_epoch:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        record_df.to_csv(save_path, index=False)

    return record_df



import matplotlib.pyplot as plt
from itertools import cycle

def learning_curve_recording(
        record_df, 
        save_path=None, 
        show=False
    ):
    metrics_list = ['AUROC', 'AUPRC']  # 고정
    num_plots = 1 + len(metrics_list)  # Loss + AUROC + AUPRC
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    # 색상 팔레트
    color_cycle = cycle(['green', 'orange', 'red', 'purple', 'cyan', 'magenta'])
    
    # === Plot 1: Loss ===
    axes[0].plot(record_df['train_loss'], label='Train Loss', color='blue', linestyle='--')
    axes[0].plot(record_df['valid_loss'], label='Valid Loss', color='blue', linestyle='-')
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # === Plot 2~: AUROC, AUPRC ===
    for cnt, met in enumerate(metrics_list):
        ax = axes[cnt + 1]

        # 클래스별 metric (열 이름 패턴 기반 탐색)
        class_cols = [col for col in record_df.columns if col.startswith(f"train_{met}_")]
        for col in class_cols:
            color = next(color_cycle)
            idx = col.split("_")[-1]  # 클래스 번호 추출
            ax.plot(record_df[col], 
                    label=f"Train Class {idx}", color=color, linestyle='--', linewidth=1, alpha=0.4)
            ax.plot(record_df[col.replace("train_", "valid_")], 
                    label=f"Valid Class {idx}", color=color, linestyle='-', linewidth=1, alpha=0.4)

        # macro_avg
        train_macro_col = f"train_macro_avg_{met}"
        valid_macro_col = f"valid_macro_avg_{met}"
        if train_macro_col in record_df.columns:
            ax.plot(record_df[train_macro_col], 
                    label=f"Train Macro {met}", color="black", linestyle='--', linewidth=2)
        if valid_macro_col in record_df.columns:
            ax.plot(record_df[valid_macro_col], 
                    label=f"Valid Macro {met}", color="black", linestyle='-', linewidth=2)

        ax.set_title(f"{met} Learning Curves")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(met)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()




def ModelSave(
        save_path, epoch, model, optimizer, 
        lr_schedular=False, amp_scaler=False
    ):
    result = {
        'epoch':None,
        'model_state_dict':None,
        'optimizer_state_dict':None,
        'lr_scheduler_state_dict':None,
        'scaler_state_dict':None,        
    }
    result['epoch'] = epoch
    result['model_state_dict'] = model.state_dict()
    result['optimizer_state_dict'] = optimizer.state_dict()
    if lr_schedular: result['lr_scheduler_state_dict'] = lr_schedular.state_dict()
    if amp_scaler: result['scaler_state_dict'] = amp_scaler.state_dict()
    torch.save(result, save_path)
    print(f"Saved at {save_path}")




import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from multiprocessing import Pool
import tqdm
import pandas as pd

def visualize_12_lead_ecg(ecg_data, fs=500, output_path=None):
    """
    12리드 ECG 데이터를 한 장으로 시각화하고 저장하는 함수.
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    if ecg_data.shape[0] != 12:
        raise ValueError("ecg_data는 12개의 리드를 포함해야 합니다 (shape: [12, n_samples]).")
    
    n_leads, n_samples = ecg_data.shape
    time = np.arange(n_samples) / fs  

    # 그림 설정
    fig, axes = plt.subplots(6, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_leads):
        axes[i].plot(time, ecg_data[i], color='blue', linewidth=0.8)
        axes[i].set_title(f"Lead {lead_names[i]}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude (mV)")
        axes[i].grid(alpha=0.5)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)  
        plt.close(fig)  # 🔥 특정 Figure 닫기 (메모리 누수 방지)
    else:
        plt.show()



import numpy as np
from scipy import signal
from tqdm import tqdm

def bandpass_filter_ecg(
    arrays, fs, cutoff_low=1, cutoff_high=40, order=5, remove_lag=True
):
    """
    Args:
        arrays: (num_leads, signal_length) numpy array
        fs: sampling frequency
        cutoff_low: low cutoff frequency (Hz)
        cutoff_high: high cutoff frequency (Hz)
        order: filter order
        remove_lag: whether to use zero-phase filtering
    Returns:
        bandpassed arrays with same shape as input
    """
    nyq = 0.5 * fs
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')

    arrays_filtered = []

    for i in range(arrays.shape[0]):  # 리드별로 필터링
        array = arrays[i, :]

        if remove_lag:
            array_filtered = signal.sosfiltfilt(sos, array)
        else:
            array_filtered = signal.sosfilt(sos, array)

        arrays_filtered.append(array_filtered)

    arrays_filtered = np.stack(arrays_filtered, axis=0)
    return arrays_filtered