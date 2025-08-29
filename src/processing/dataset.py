from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from scipy.signal import resample
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
import pandas as pd
from typing import Iterable, Literal, Optional
import os
import pickle as pkl

class DatasetFromDataframe(Dataset):
    _LEAD_SLICE = {"12lead": slice(0, 12),
                   "limb_lead": slice(0, 6),
                   "lead1": slice(0, 1),
                   "lead2": slice(1, 2)}
    
    def __init__(self,
                 df: pd.DataFrame, 
                 input_col: str, 
                 label_col: Optional[Iterable] = None,
                 target_lead: str = "12lead",
                 target_len: int = 2560,
                 transforms: Optional[object] = None,
                 label_transforms: Optional[object] = None,):
        
        """
        Args:
            ...
        """        
        self.df = df
        self.input_path = df[input_col].values
        self.label_col = label_col
        if self.label_col:
            self.label = df[self.label_col].values
        self.target_lead = target_lead
        self.target_len = target_len
        self.transforms = transforms
        self.label_transforms = label_transforms

    def check_dataset(self, x):
        assert x.shape[0] < x.shape[1], \
            f"maybe need to transpose, x.shape = {x.shape}"
        assert self.target_lead in self._LEAD_SLICE.keys(), \
            f"target_lead should be one of {list(self._LEAD_SLICE.keys())}"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        fpath = self.input_path[idx]
        _, ext = os.path.splitext(fpath) ; ext = ext.lower()
        if ext == '.pkl':
            with open(fpath, 'rb') as f:
                x = pkl.load(f)  # (leads, seq_len)
        elif ext == '.npy':
            x = np.load(fpath)  # (leads, seq_len)
        else:
            raise ValueError(f"Unsupported file type: {ext}")  
                  
        lead_slice = self._LEAD_SLICE[self.target_lead]
        x = x[lead_slice, :]
        self.check_dataset(x)

        if x.shape[-1] != self.target_len:
            x = resample(x, self.target_len, axis=-1)
            x = x.astype(np.float32)

        if self.transforms:
            x = self.transforms(x)

        output = {'input': torch.tensor(x, dtype=torch.float32),
                  'input_path': self.input_path[idx],}
        if self.label_col:
            y = self.label[idx]
            if self.label_transforms:
                y = self.label_transforms(y)
            else:
                y = torch.tensor(y, dtype=torch.float32)
            # output['label'] = y
            output['label'] = y.squeeze(-1)

        return output
        

class DatasetFromDataframe_v2(Dataset):
    _LEAD_SLICE = {"12lead": slice(0, 12),
                   "limb_lead": slice(0, 6),
                   "lead1": slice(0, 1),
                   "lead2": slice(1, 2)}
    
    def __init__(self,
                 df: pd.DataFrame, 
                 input_col: str, 
                 label_col: Optional[Iterable] = None,
                 target_lead: str = "12lead",
                 target_len: int = 2560,
                 transforms: Optional[object] = None,
                 label_transforms: Optional[object] = None,):
        
        """
        Args:
            ...
        """        
        self.df = df
        self.input_path = df[input_col].values
        self.label_col = label_col
        if self.label_col:
            self.label = df[self.label_col].values
        self.target_lead = target_lead
        self.target_len = target_len
        self.transforms = transforms
        self.label_transforms = label_transforms

    def check_dataset(self, x):
        assert x.shape[0] < x.shape[1], \
            f"maybe need to transpose, x.shape = {x.shape}"
        assert self.target_lead in self._LEAD_SLICE.keys(), \
            f"target_lead should be one of {list(self._LEAD_SLICE.keys())}"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        fpath = self.input_path[idx]
        _, ext = os.path.splitext(fpath) ; ext = ext.lower()
        if ext == '.pkl':
            with open(fpath, 'rb') as f:
                x = pkl.load(f)  # (leads, seq_len)
        elif ext == '.npy':
            x = np.load(fpath)  # (leads, seq_len)
        else:
            raise ValueError(f"Unsupported file type: {ext}")  
                  
        lead_slice = self._LEAD_SLICE[self.target_lead]
        x = x[lead_slice, :]
        self.check_dataset(x)

        if x.shape[-1] != self.target_len:
            x = resample(x, self.target_len, axis=-1)
            x = x.astype(np.float32)
            if self.target_len != 2560:
                x = x[:, 256:2560-256]
            mu = np.median(x)          # 평균 대신 median 권장
            sigma = np.std(np.clip(x, np.percentile(x,1), np.percentile(x,99)))  # 이상치 완화
            x = (x - mu) / (sigma + 1e-8)
        if self.transforms:
            x = self.transforms(x)

        output = {'input': torch.tensor(x, dtype=torch.float32),
                  'input_path': self.input_path[idx],}
        if self.label_col:
            y = self.label[idx]
            if self.label_transforms:
                y = self.label_transforms(y)
            else:
                y = torch.tensor(y, dtype=torch.float32)
            # output['label'] = y
            output['label'] = y.squeeze(-1)

        return output


import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from typing import Optional, Iterable, Dict, Any, Callable, List

# --- 최종 개선된 ECG Dataset 클래스 ---
class ECGDataset(Dataset):
    """
    다양한 소스(파일 경로 리스트, 데이터프레임 등)로부터 ECG 데이터셋을 생성할 수 있는
    유연한 PyTorch Dataset 클래스입니다.
    """
    _LEAD_SLICE = {
        "12lead": slice(0, 12),
        "limb_lead": slice(0, 6),
        "lead1": slice(0, 1),
        "lead2": slice(1, 2)
    }
    
    # --- 핵심 로직: 스태틱 메소드 (v3와 동일) ---
    @staticmethod
    def load_ecg(fpath: str) -> np.ndarray:
        # ... (이전과 동일) ...
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
        if ext == '.pkl':
            with open(fpath, 'rb') as f:
                return pkl.load(f)
        elif ext == '.npy':
            return np.load(fpath)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    @staticmethod
    def preprocess_ecg(
        x: np.ndarray,
        target_lead: str,
        target_len: int,
        do_resample: bool = True,
        do_normalize: bool = True,
        transforms: Optional[Callable] = None,
    ) -> np.ndarray:
        # ... (이전과 동일) ...
        assert target_lead in ECGDataset._LEAD_SLICE, \
            f"target_lead는 다음 중 하나여야 합니다: {list(ECGDataset._LEAD_SLICE.keys())}"
        lead_slice = ECGDataset._LEAD_SLICE[target_lead]
        x = x[lead_slice, :]
        assert x.shape[0] < x.shape[1], \
            f"데이터 형식이 (seq_len, leads)일 수 있습니다. (leads, seq_len)으로 변경해야 합니다. 현재 shape: {x.shape}"
        if do_resample and x.shape[-1] != target_len:
            x = resample(x, target_len, axis=-1).astype(np.float32)
        if do_normalize:
            clipped_x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
            mu = np.median(x, axis=-1, keepdims=True)
            sigma = np.std(clipped_x, axis=-1, keepdims=True)
            x = (x - mu) / (sigma + 1e-8)
        if transforms:
            x = transforms(x)
        return x.astype(np.float32)

    # --- 기본 생성자: 이제 파일 경로 리스트를 직접 받음 ---
    def __init__(self,
                 file_paths: List[str],
                 labels: Optional[List[Any]] = None,
                 target_lead: str = "12lead",
                 target_len: int = 2560,
                 do_resample: bool = True,
                 do_normalize: bool = True,
                 transforms: Optional[Callable] = None,
                 label_transforms: Optional[Callable] = None):
        
        self.file_paths = file_paths
        self.labels = labels
        
        if self.labels is not None:
            assert len(self.file_paths) == len(self.labels), "입력(file_paths)과 레이블(labels)의 개수가 일치해야 합니다."
        
        self.target_lead = target_lead
        self.target_len = target_len
        self.do_resample = do_resample
        self.do_normalize = do_normalize
        self.transforms = transforms
        self.label_transforms = label_transforms
    
    # --- 클래스 메소드 팩토리: 데이터프레임으로부터 클래스 인스턴스 생성 ---
    @classmethod
    def from_dataframe(cls, 
                       df: pd.DataFrame, 
                       input_col: str, 
                       label_col: Optional[Iterable] = None,
                       **kwargs):
        """
        Pandas DataFrame에서 파일 경로와 레이블을 추출하여
        ECGDataset 인스턴스를 생성하는 팩토리 메소드입니다.
        """
        file_paths = df[input_col].values.tolist()
        labels = df[label_col].values.tolist() if label_col else None
        
        # cls는 ECGDataset 클래스 자신을 의미합니다.
        # **kwargs를 통해 target_lead, target_len 등의 추가 인자를 전달합니다.
        return cls(file_paths=file_paths, labels=labels, **kwargs)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self.file_paths[idx]
        x = self.load_ecg(fpath)
        x = self.preprocess_ecg(
            x=x,
            target_lead=self.target_lead,
            target_len=self.target_len,
            do_resample=self.do_resample,
            do_normalize=self.do_normalize,
            transforms=self.transforms,
        )
        output = {'input': torch.from_numpy(x), 'input_path': fpath}
        
        if self.labels is not None:
            y = self.labels[idx]
            if self.label_transforms:
                y = self.label_transforms(y)
            y = torch.as_tensor(y, dtype=torch.float32)
            output['label'] = y.squeeze()

        return output


from processing import utils

class ECGDataset_v2(Dataset):

    '''
        1. * 나중에는 xml을 입력으로 받아서 처리할 수 있어야함.
        2. (가정) xml을 입력으로 받아서 아래와 같은 데이터 형태가 있어야함.
            {'signal'         : array, 
             'fs'             : fs, 
             'scaling_factor' : scaling_factor}
    '''

    @staticmethod
    def load_ecg_from_xml():
        '''
            XML 파일에서 ECG 신호와 메타데이터를 로드하는 메소드입니다.
            기대되는 입력은 ...
        '''
        pass


    @staticmethod
    def preprocess_ecg(
        x: np.ndarray,
        scaling_factor: Optional[float] = None,
        do_bandpass: dict = None,
        original_fs: int = None,
        target_fs: int = 256,
        do_normalize: bool = False,
        transforms: Optional[Callable] = None,
    ) -> np.ndarray:
        # 1. check
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")
        if x.shape[0] != 12:
            raise ValueError(f"Input data does not have 12 leads, got {x.shape[0]} leads.")
        if original_fs is not None and x.shape[1] != original_fs * 10:
            raise ValueError(f"Input data is not 10 seconds long. Expected {original_fs*10} samples, got {x.shape[1]} samples.")
        # 2. scaling
        if scaling_factor is not None:
            x = x * scaling_factor
        # 3. bandpass filtering
        if do_bandpass is not None:
            from processing import utils
            x = utils.bandpass_filter_ecg(x, original_fs, cutoff_low=do_bandpass['lowcut'], cutoff_high=do_bandpass['highcut'], order=do_bandpass['order'])
        # 4. resample
        if x.shape[-1] != target_fs:
            x = resample(x, int(target_fs*10), axis=-1).astype(np.float32)
        # 5. normalize
        if do_normalize:
            clipped_x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
            mu = np.median(x, axis=-1, keepdims=True)
            sigma = np.std(clipped_x, axis=-1, keepdims=True)
            x = (x - mu) / (sigma + 1e-8)
        # 6. transforms
        if transforms:
            x = transforms(x)
        return x.astype(np.float32)
    

    @staticmethod
    def load_ecg(fpath: str) -> np.ndarray:
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
        if ext == '.pkl':
            with open(fpath, 'rb') as f:
                return pkl.load(f)
        elif ext == '.npy':
            return np.load(fpath)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


    # --- 기본 생성자: 이제 파일 경로 리스트를 직접 받음 ---
    def __init__(self,
                 file_paths: List[str],
                 labels: Optional[List[Any]] = None,
                 target_lead: str = "12lead",
                 target_len: int = 2560,
                 do_resample: bool = True,
                 do_normalize: bool = True,
                 transforms: Optional[Callable] = None,
                 label_transforms: Optional[Callable] = None):
        
        self.file_paths = file_paths
        self.labels = labels
        
        if self.labels is not None:
            assert len(self.file_paths) == len(self.labels), "입력(file_paths)과 레이블(labels)의 개수가 일치해야 합니다."
        
        self.target_lead = target_lead
        self.target_len = target_len
        self.do_resample = do_resample
        self.do_normalize = do_normalize
        self.transforms = transforms
        self.label_transforms = label_transforms
    
    # --- 클래스 메소드 팩토리: 데이터프레임으로부터 클래스 인스턴스 생성 ---
    @classmethod
    def from_dataframe(cls, 
                       df: pd.DataFrame, 
                       input_col: str, 
                       label_col: Optional[Iterable] = None,
                       **kwargs):
        """
        Pandas DataFrame에서 파일 경로와 레이블을 추출하여
        ECGDataset 인스턴스를 생성하는 팩토리 메소드입니다.
        """
        file_paths = df[input_col].values.tolist()
        labels = df[label_col].values.tolist() if label_col else None
        
        # cls는 ECGDataset 클래스 자신을 의미합니다.
        # **kwargs를 통해 target_lead, target_len 등의 추가 인자를 전달합니다.
        return cls(file_paths=file_paths, labels=labels, **kwargs)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self.file_paths[idx]
        x = self.load_ecg(fpath)
        x = self.preprocess_ecg(
            x=x,
            target_lead=self.target_lead,
            target_len=self.target_len,
            do_resample=self.do_resample,
            do_normalize=self.do_normalize,
            transforms=self.transforms,
        )
        output = {'input': torch.from_numpy(x), 'input_path': fpath}
        
        if self.labels is not None:
            y = self.labels[idx]
            if self.label_transforms:
                y = self.label_transforms(y)
            y = torch.as_tensor(y, dtype=torch.float32)
            output['label'] = y.squeeze()

        return output























