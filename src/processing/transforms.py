# transform_pipeline = _kimbg_transforms.TransformPipeline([
    # _kimbg_transforms.NormalizeECG(method="tanh", scope="lead-wise", scale=2), 
    # _kimbg_transforms.RandomLeadMasking(mask_ratio=0.2, mask_leads=None, p=p),
    # _kimbg_transforms.RandomLeadSwapping(swap_ratio=0.2, swap_pairs=None, p=p),

    # _kimbg_transforms.ApplyGaussianNoise(noise_level=0.5, mode='add', per_lead=True, p=p), 
    # _kimbg_transforms.ApplyGaussianNoise(noise_level=0.5, mode='mul', per_lead=True, p=p),  
    # _kimbg_transforms.ApplyGaussianNoise(noise_level=0.5, mode='add', per_lead=False, p=p), 
    # _kimbg_transforms.ApplyGaussianNoise(noise_level=0.5, mode='mul', per_lead=False, p=p),  

    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(0, 2), amplitude_range=(0, 0.1), mode="add", per_lead=True, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(2, 5), amplitude_range=(0, 0.05), mode="add", per_lead=True, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(5, 50), amplitude_range=(0, 0.02), mode="add", per_lead=True, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(0, 2), amplitude_range=(0, 0.1), mode="add", per_lead=False, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(2, 5), amplitude_range=(0, 0.05), mode="add", per_lead=False, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(5, 50), amplitude_range=(0, 0.02), mode="add", per_lead=False, p=p
    # ),

    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(0, 50), amplitude_range=(0, 0.5), mode="mul", per_lead=True, p=p
    # ),
    # _kimbg_transforms.ApplySinusoidalNoise(
    #     frequency_range=(0, 50), amplitude_range=(0, 0.5), mode="mul", per_lead=False, p=p
    # ),

    # _kimbg_transforms.RandomZeroMasking(mask_ratio=0.2, per_lead=True, p=p),
    # _kimbg_transforms.AddTimeWarp(warp_factor_range=(0.9, 1.1), per_lead=False, padding_mode="repeat", p=p),        
# ])

import numpy as np
import random

def replace_outliers_with_median(data, low=-5, high=5):
    median = np.median(data)
    data = np.where((data >= low) & (data <= high), data, median)
    return data
class OutlierReplace:
    def __init__(self, low=-5, high=5, methods='zero'):
        self.low = low
        self.high = high
        self.methods = methods
    def __call__(self, ecg_data):
        if self.methods == 'zero':
            return np.where((ecg_data >= self.low) & (ecg_data <= self.high), ecg_data, 0)
        elif self.methods == 'median':
            return replace_outliers_with_median(ecg_data, low=self.low, high=self.high)
        elif self.methods == 'clip':
            return np.clip(ecg_data, self.low, self.high)
        else:
            raise ValueError("methods는 'zero', 'median', 'clip' 중 하나여야 합니다.")            


class SliceSignal:
    """
    시퀀스를 target_length로 잘라주는 Transform 클래스.
    예) seq_length=2500 -> target_length=2400이면
        왼쪽에서 50, 오른쪽에서 50 잘라 2400 길이로 반환.
    """
    def __init__(self, target_length):
        self.target_length = target_length

    def __call__(self, ecg_data):
        """
        ecg_data: 1D (length,) 또는 2D (leads, length) 형태 가정.
        """
        # 현재 길이
        if ecg_data.ndim == 1:
            current_length = ecg_data.shape[0]
        elif ecg_data.ndim == 2:
            current_length = ecg_data.shape[1]
        else:
            raise ValueError("ecg_data는 최대 2차원(리드 수, 길이) 형태여야 합니다.")
        
        if current_length < self.target_length:
            raise ValueError(f"현재 시퀀스 길이({current_length})가 target_length({self.target_length})보다 작습니다.")
        elif current_length == self.target_length:
            # 자를 필요가 없음
            return ecg_data
        
        # 자를 길이(difference)
        diff = current_length - self.target_length
        # 왼쪽과 오른쪽에서 얼마나 자를지
        left_cut = diff // 2
        right_cut = diff - left_cut  # diff가 홀수일 경우 오른쪽이 한 샘플 더 많아질 수 있음

        if ecg_data.ndim == 1:
            # 1D: (length,)
            return ecg_data[left_cut : current_length - right_cut]
        else:
            # 2D: (leads, length)
            return ecg_data[:, left_cut : current_length - right_cut]


def normalize_ecg(ecg_data, method="z-score", scope="lead-wise", scale=1):
    """
    ECG 데이터 정규화 수행 (리드별 또는 전체 범위 기준).

    Parameters:
    - ecg_data: np.array, shape (num_leads, num_samples)
        다중 리드 ECG 데이터
    - method: str, "z-score" | "min-max" | "max-abs" | "tanh"
        정규화 방식
        - "z-score": 평균을 0, 표준편차를 1로 정규화
        - "min-max": 최소값 0, 최대값 1로 정규화
        - "max-abs": 최대 절대값 기준으로 정규화
        - "tanh": 하이퍼볼릭 탄젠트 정규화
    - scope: str, "lead-wise" | "global"
        정규화 범위 선택
        - "lead-wise": 리드별 정규화
        - "global": 전체 데이터 기준 정규화

    Returns:
    - normalized_data: np.array, 동일한 shape
    """
    
    # EUMC dataset (392k) 기준 z-norm을 위한 각 리드별 평균과 표준편차. (raw signal)
    means = [
        -0.00984134, -0.00618936, 0.0005158, 0.01029302, -0.00596927, -0.00436603,
        0.00369437, -0.00168305, -0.00173444, -0.00383145, -0.00424788, -0.00555575,
    ]
    stds = [
        0.15452304, 0.1761096,  0.15970715, 0.14569322, 0.13040085, 0.15050802,
        0.29022712, 0.36810402, 0.38695719, 0.44395544, 0.46174094, 0.54878569,        
    ]

    if scope == "lead-wise":
        normalized_data = []
        for idx, lead in enumerate(ecg_data):
            if method == "z-score":
                # normalized_lead = (lead - np.mean(lead)) / np.std(lead)
                normalized_lead = (lead - means[idx]) / stds[idx]
            elif method == "min-max":
                normalized_lead = (lead - np.min(lead)) / (np.max(lead) - np.min(lead) + 1e-9)
            elif method == "max-abs":
                normalized_lead = lead / (np.max(np.abs(lead)) + 1e-9)
            elif method == "tanh":
                normalized_lead = np.tanh(lead * scale)
            else:
                raise ValueError("지원하지 않는 정규화 방식입니다. method는 'z-score', 'min-max', 'max-abs', 'tanh' 중 하나여야 합니다.")
            normalized_data.append(normalized_lead)
        return np.array(normalized_data)

    elif scope == "global":
        if method == "z-score":
            # normalized_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
            normalized_data = (ecg_data - np.mean(means)) / np.mean(stds)
        elif method == "min-max":
            normalized_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
        elif method == "max-abs":
            normalized_data = ecg_data / np.max(np.abs(ecg_data))
        elif method == "tanh":
            normalized_data = np.tanh(ecg_data * scale) 
        else:
            raise ValueError("지원하지 않는 정규화 방식입니다. method는 'z-score', 'min-max', 'max-abs', 'tanh' 중 하나여야 합니다.")
        return normalized_data

    else:
        raise ValueError("scope는 'lead-wise' 또는 'global' 중 하나여야 합니다.")
class NormalizeECG:
    def __init__(self, method="z-score", scope="lead-wise", scale=1):
        self.method = method
        self.scope = scope
        self.scale = scale
    def __call__(self, ecg_data):
        return normalize_ecg(ecg_data, method=self.method, scope=self.scope, scale=self.scale)


def apply_gaussian_noise(signal, noise_level=0.1, mode="add", per_lead=True):
    """
    Apply Gaussian noise to the signal, either additively or multiplicatively, 
    with an option to apply per lead or uniformly across all leads.
    
    Args:
        signal (np.array): Input signal (1D or 2D array).
        noise_level (float): Noise intensity (standard deviation relative to signal range).
        mode (str): "add" for additive noise, "mul" for multiplicative noise.
        per_lead (bool): If True, generate independent noise for each lead/channel.
    
    Returns:
        np.array: Signal with Gaussian noise applied.
    """
    if signal.ndim == 1:  # 1D 신호
        if mode == "add":
            noise = np.random.normal(0, noise_level * np.std(signal), size=signal.shape)
            return signal + noise
        elif mode == "mul":
            noise = np.random.normal(1, noise_level, size=signal.shape)
            return signal * noise
        else:
            raise ValueError("Invalid mode. Choose 'add' or 'mul'.")
    
    elif signal.ndim == 2:  # 2D 신호 (다채널)
        channels, samples = signal.shape
        if mode == "add":
            if per_lead:  # 리드별 독립적 노이즈
                noise = np.random.normal(0, noise_level * np.std(signal, axis=1, keepdims=True), size=signal.shape)
            else:  # 모든 리드에 동일한 노이즈
                noise = np.random.normal(0, noise_level * np.std(signal), size=(1, samples))
            return signal + noise
        
        elif mode == "mul":
            if per_lead:  # 리드별 독립적 노이즈
                noise = np.random.normal(1, noise_level, size=signal.shape)
            else:  # 모든 리드에 동일한 노이즈
                noise = np.random.normal(1, noise_level, size=(1, samples))
            return signal * noise
        
        else:
            raise ValueError("Invalid mode. Choose 'add' or 'mul'.")
    
    else:
        raise ValueError("Input signal must be 1D or 2D array.")
class ApplyGaussianNoise:
    def __init__(self, noise_level=0.1, mode='add', per_lead=False, p=0.5):
        self.noise_level = noise_level
        self.mode = mode
        self.per_lead = per_lead
        self.probability = p
    def __call__(self, signal):
        if np.random.rand() < self.probability:  # 확률에 따라 적용
            # print('ApplyGaussianNoise')
            return apply_gaussian_noise(signal, noise_level=self.noise_level, mode=self.mode, per_lead=self.per_lead)
        return signal
    

def apply_sinusoidal_noise(signal, frequency_range=(0.5, 2), amplitude_range=(0.01, 0.05), fs=250, mode="add", per_lead=True):
    """
    Apply sinusoidal noise to the signal.
    Option guide:
        sensitive signal like ecg -> frequency_range=(0.5, 2), amplitude_range=(0.01, 0.05)
        general signal -> frequency_range=(1, 5), amplitude_range=(0.05, 0.2)
        extreme signal -> frequency_range=(0.1, 10), amplitude_range=(0.2, 1)
    
    Args:
        signal (np.array): Input signal (1D or 2D array: [channels, samples]).
        frequency_range (tuple): Min and max frequencies of sinusoidal noise (Hz).
        amplitude_range (tuple): Min and max amplitudes of sinusoidal noise.
        fs (int): Sampling frequency (Hz).
        mode (str): "add" for additive noise, "mul" for multiplicative noise.
        per_lead (bool): If True, generate independent noise for each lead/channel.
        
    Returns:
        np.array: Signal with sinusoidal noise applied.
    """
    length = signal.shape[-1]  # 신호 길이
    t = np.arange(length) / fs  # 시간 벡터 생성

    if signal.ndim == 1:  # 1D 신호
        frequency = np.random.uniform(*frequency_range)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(0, 2 * np.pi)
        sinusoid = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        if mode == "add":
            return signal + sinusoid
        elif mode == "mul":
            return signal * (1 + sinusoid)
        else:
            raise ValueError("Invalid mode. Choose 'add' or 'mul'.")

    elif signal.ndim == 2:  # 2D 신호 (다채널)
        channels, samples = signal.shape
        sinusoid_noise = np.zeros_like(signal)

        for ch in range(channels):
            if per_lead or ch == 0:  # 리드별 독립적 노이즈 생성
                frequency = np.random.uniform(*frequency_range)
                amplitude = np.random.uniform(*amplitude_range)
                phase = np.random.uniform(0, 2 * np.pi)
            sinusoid = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            sinusoid_noise[ch, :] = sinusoid

        if mode == "add":
            return signal + sinusoid_noise
        elif mode == "mul":
            return signal * (1 + sinusoid_noise)
        else:
            raise ValueError("Invalid mode. Choose 'add' or 'mul'.")
    else:
        raise ValueError("Input signal must be 1D or 2D array.")
class ApplySinusoidalNoise:
    def __init__(self, frequency_range=(0.5, 2), amplitude_range=(0.01, 0.05), fs=250, mode="add", per_lead=True, p=0.5):
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.fs = fs
        self.mode = mode
        self.per_lead = per_lead
        self.probability = p
    def __call__(self, signal):
        if np.random.rand() < self.probability:  # 확률에 따라 적용
            # print('ApplySinusoidalNoise')
            return apply_sinusoidal_noise(
                signal, frequency_range=self.frequency_range, amplitude_range=self.amplitude_range, 
                fs=self.fs, mode=self.mode, per_lead=self.per_lead,
            )
        return signal
    

def random_zero_masking(signal, mask_ratio=0.2, per_lead=False):
    """
    Randomly mask a segment of the signal with zeros, preserving the original length.
    
    Args:
        signal (np.array): Input signal (1D or 2D array: [channels, samples]).
        slice_ratio (float): Ratio of the slice length to the signal length (0 < slice_ratio <= 1).
        per_lead (bool): If True, mask each lead independently. If False, apply the same mask across all leads.
    
    Returns:
        np.array: Signal with the specified segment masked by zeros.
    """
    signal = signal.copy()
    original_length = signal.shape[-1]

    # 슬라이싱 길이 계산
    if not (0 < mask_ratio <= 1):
        raise ValueError("slice_ratio must be between 0 and 1.")
    slice_length = int(original_length * mask_ratio)

    if signal.ndim == 1:  # 1D 신호
        # 슬라이싱 시작점 결정
        start = random.randint(0, original_length - slice_length)
        signal[start:start + slice_length] = 0  # 해당 구간에 0 값 삽입
        return signal

    elif signal.ndim == 2:  # 2D 신호 (다채널)
        channels, samples = signal.shape

        if per_lead:  # 각 리드를 독립적으로 마스킹
            for ch in range(channels):
                start = random.randint(0, samples - slice_length)
                signal[ch, start:start + slice_length] = 0  # 해당 구간에 0 값 삽입
        else:  # 모든 리드에 동일한 구간 마스킹
            start = random.randint(0, samples - slice_length)
            signal[:, start:start + slice_length] = 0  # 해당 구간에 0 값 삽입

        return signal

    else:
        raise ValueError("Input signal must be 1D or 2D array.")
class RandomZeroMasking:
    def __init__(self, mask_ratio=0.2, per_lead=False, p=0.5):
        self.mask_ratio = mask_ratio
        self.per_lead = per_lead
        self.probability = p
    def __call__(self, signal):
        if np.random.rand() < self.probability:  # 확률에 따라 적용
            # print('RandomZeroMasking')
            return random_zero_masking(signal, mask_ratio = self.mask_ratio, per_lead = self.per_lead,)
        return signal



def time_warp(signal_, warp_factor_range=(0.9, 1.1), padding_mode="repeat"):
    """
    Apply time warping to a 1D or 2D signal (e.g. multi-lead ECG) and
    then adjust it to its original shape.
    
    - 1D (shape: (samples,)):
      A single-channel signal is warped with a random factor in warp_factor_range.
    - 2D (shape: (channels, samples)):
      Each channel is warped with the same warp_factor so that
      all leads remain time-synchronized (no per_lead argument).
    
    Args:
        signal_ (np.array): Input signal.
            - 1D shape: (samples,)
            - 2D shape: (channels, samples)
        warp_factor_range (tuple): (min_factor, max_factor) time warp range.
        padding_mode (str): 
            - "zero"   -> zero-pad 
            - "repeat" -> wrap-around padding

    Returns:
        np.array: Time-warped signal (same shape as input).
    """
    signal = signal_.copy()

    if signal.ndim == 1:
        # 1D: 단일 채널
        target_length = len(signal)
        warp_factor = np.random.uniform(*warp_factor_range)
        # 시간 축 생성
        indices = np.linspace(0, target_length - 1, int(target_length * warp_factor))
        # 보간
        warped_1d = np.interp(indices, np.arange(target_length), signal)
        # 길이 보정
        warped_1d = _adjust_length(warped_1d, target_length, padding_mode)
        return warped_1d

    elif signal.ndim == 2:
        # 2D: 다채널 (channels, samples)
        channels, target_length = signal.shape
        warped_signal = np.zeros((channels, target_length))

        # 모든 리드(채널)에 동일한 warp_factor 적용
        warp_factor = np.random.uniform(*warp_factor_range)
        indices = np.linspace(0, target_length - 1, int(target_length * warp_factor))

        for ch in range(channels):
            # 각 리드는 "자기" 신호를 동일 warp_factor로 보간
            warped_ch = np.interp(indices, np.arange(target_length), signal[ch, :])
            warped_ch = _adjust_length(warped_ch, target_length, padding_mode)
            warped_signal[ch, :] = warped_ch

        return warped_signal

    else:
        raise ValueError("Input signal must be 1D or 2D array.")
def _adjust_length(warped_1d, target_length, padding_mode):
    """타임워핑 후 길이가 달라진 1D 신호를 target_length에 맞춰서 잘라내거나 패딩."""
    current_len = len(warped_1d)

    if current_len > target_length:
        # 너무 길면 중앙을 기준으로 자름
        start_idx = (current_len - target_length) // 2
        warped_1d = warped_1d[start_idx:start_idx + target_length]

    elif current_len < target_length:
        # 길이가 짧으면 패딩
        pad_length = target_length - current_len
        if padding_mode == "zero":
            warped_1d = np.pad(warped_1d, (0, pad_length), mode='constant')
        elif padding_mode == "repeat":
            warped_1d = np.pad(warped_1d, (0, pad_length), mode='wrap')

    return warped_1d
class AddTimeWarp:
    """
    Transform-like class that applies time_warp with a given probability p.
    (No per_lead option; multi-lead ECG stays time-synchronized.)
    """
    def __init__(self, warp_factor_range=(0.9, 1.1), padding_mode='repeat', p=0.5):
        self.warp_factor_range = warp_factor_range
        self.padding_mode = padding_mode
        self.probability = p

    def __call__(self, signal):
        if np.random.rand() < self.probability:
            return time_warp(
                signal,
                warp_factor_range=self.warp_factor_range,
                padding_mode=self.padding_mode
            )
        return signal


def random_lead_masking(signal_, mask_ratio=0.5, mask_leads=None):
    """
    Randomly masks entire leads in an ECG signal.

    Args:
        signal_ (np.array): Input ECG signal (2D array: [leads, samples]).
        mask_ratio (float): Probability of masking a lead (0 ~ 1).
        mask_leads (list of int): Specific leads to mask. If None, randomly selected.

    Returns:
        np.array: ECG signal with masked leads.
    """
    signal = signal_.copy()
    n_leads, _ = signal.shape

    if mask_leads is None:
        # 마스킹할 리드를 랜덤하게 선택
        n_masked_leads = int(n_leads * mask_ratio)
        mask_leads = random.sample(range(n_leads), n_masked_leads)

    for lead in mask_leads:
        signal[lead, :] = 0  # 해당 리드 전체를 0으로 마스킹

    return signal
class RandomLeadMasking:
    def __init__(self, mask_ratio=0.5, mask_leads=None, p=0.5):
        """
        ECG 신호에서 랜덤 리드 마스킹을 수행하는 클래스.

        Args:
            mask_ratio (float): 랜덤하게 마스킹할 리드의 비율 (0 ~ 1).
            mask_leads (list of int): 특정 리드를 마스킹할 경우 지정 (None이면 랜덤 선택).
            p (float): 마스킹을 적용할 확률 (0 ~ 1).
        """
        self.mask_ratio = mask_ratio
        self.mask_leads = mask_leads
        self.probability = p

    def __call__(self, signal):
        """
        ECG 신호에 랜덤 리드 마스킹을 적용.

        Args:
            signal (np.array): Input ECG signal (2D array: [leads, samples]).

        Returns:
            np.array: 마스킹된 ECG 신호.
        """
        if np.random.rand() < self.probability:  # 확률에 따라 적용
            return random_lead_masking(signal, mask_ratio=self.mask_ratio, mask_leads=self.mask_leads)
        return signal


def random_lead_swapping(signal_, swap_ratio=0.5, swap_pairs=None):
    """
    Randomly swaps ECG leads to augment data.

    Args:
        signal_ (np.array): Input ECG signal (2D array: [leads, samples]).
        swap_ratio (float): Probability of swapping a lead pair (0 ~ 1).
        swap_pairs (list of tuples): Specific lead pairs to swap (e.g., [(0, 1), (2, 3)]).
                                     If None, pairs are randomly chosen.

    Returns:
        np.array: ECG signal with swapped leads.
    """
    signal = signal_.copy()
    n_leads, _ = signal.shape

    # Swap lead pairs
    if swap_pairs is None:
        all_leads = list(range(n_leads))
        random.shuffle(all_leads)
        swap_pairs = [(all_leads[i], all_leads[i+1]) for i in range(0, len(all_leads)-1, 2)]
    
    for lead1, lead2 in swap_pairs:
        if random.random() < swap_ratio:  # swap_ratio 확률로 스와핑 수행
            signal[lead1], signal[lead2] = signal[lead2].copy(), signal[lead1].copy()

    return signal
class RandomLeadSwapping:
    def __init__(self, swap_ratio=0.5, swap_pairs=None, p=0.5):
        """
        ECG 신호에서 랜덤 리드 스와핑을 수행하는 클래스.

        Args:
            swap_ratio (float): 스와핑할 리드 쌍의 확률 (0 ~ 1).
            swap_pairs (list of tuples): 특정 리드 쌍을 스와핑할 경우 지정 (None이면 랜덤 선택).
            p (float): 스와핑을 적용할 확률 (0 ~ 1).
        """
        self.swap_ratio = swap_ratio
        self.swap_pairs = swap_pairs
        self.probability = p

    def __call__(self, signal):
        """
        ECG 신호에 랜덤 리드 스와핑을 적용.

        Args:
            signal (np.array): Input ECG signal (2D array: [leads, samples]).

        Returns:
            np.array: 리드가 랜덤하게 스와핑된 ECG 신호.
        """
        if np.random.rand() < self.probability:  # 확률에 따라 적용
            return random_lead_swapping(signal, swap_ratio=self.swap_ratio, swap_pairs=self.swap_pairs)
        return signal


# Transform pipeline 클래스는 그대로 유지
class TransformPipeline:
    def __init__(self, transforms):
        """
        Args:
            transforms (list of callables): A list of transform objects.
        """
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        # print('-'*20)
        return data


import torch
from typing import Any, Dict, List, Optional, Tuple, Union
class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)