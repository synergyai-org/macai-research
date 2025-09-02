from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
import pandas as pd
from typing import Iterable, Optional, Callable, List, Any, Dict
import os
import pickle as pkl
from parsing import parse_xml
from processing import utils
import pickle



class ECGDataset(Dataset):
    LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    XML_API_URL = "http://localhost:18003/parse/parse_xml"


    @staticmethod
    def _process_raw_ecg(
        signal: Dict[str, List[float]], 
        metadata: Dict[str, Any], 
        fpath: str,
        do_bandpass: Optional[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        [공통 헬퍼 메소드]
        로드된 raw signal과 metadata를 받아 공통 전처리(정렬, 확인, 클리핑 등)를 수행합니다.
        """
        try:
            # 1. 리드 순서 정렬 (버그 수정 및 예외 처리 강화)
            ordered_signal_list = [signal[lead] for lead in ECGDataset.LEAD_ORDER]
            signal_arr = np.array(ordered_signal_list, dtype=np.float32)
        except KeyError as e:
            print(f"오류: 데이터에 필요한 리드({e})가 없습니다. 파일: {os.path.basename(fpath)}")
            return None

        # 2. 데이터 형태 확인
        if signal_arr.shape[0] != 12:
            print(f"경고: 데이터의 리드 수가 12개가 아닙니다 ({signal_arr.shape[0]}개). 파일: {os.path.basename(fpath)}")
            return None

        # 3. 10초 클리핑
        original_fs = int(metadata.get('rate', 0))
        if not original_fs:
            print(f"경고: 샘플링 레이트('rate') 정보가 없습니다. 파일: {os.path.basename(fpath)}")
            return None
        
        expected_len = original_fs * 10
        original_len = signal_arr.shape[1]

        if original_len > expected_len: # 10초보다 길면 중앙을 기준으로 자르기
            start_idx = (original_len - expected_len) // 2
            signal_arr = signal_arr[:, start_idx : start_idx + expected_len]
            # signal_arr = signal_arr[:, :expected_len]
        elif original_len < expected_len:
            print(f"경고: 데이터 길이가 10초 미만입니다 ({original_len/original_fs:.2f}초). 파일: {os.path.basename(fpath)}")
            # 필요시 패딩 로직 추가 가능
            return None

        # 4. 스케일링 (안전하게 .get() 사용)
        scale_factor_val = metadata.get('scale_factor')
        if scale_factor_val is not None:
            scale_factor = float(scale_factor_val) / 1000.0
            signal_arr = signal_arr * scale_factor

        # 5. Bandpass 필터링
        if do_bandpass is not None:
            signal_arr = utils.bandpass_filter_ecg(
                signal_arr, original_fs, 
                cutoff_low=do_bandpass['lowcut'], 
                cutoff_high=do_bandpass['highcut'], 
                order=do_bandpass['order']
            )
        
        return signal_arr

    @staticmethod
    def load_ecg_from_xml(fpath: str, do_bandpass: dict = None, api_url: str = XML_API_URL) -> Optional[np.ndarray]:
        """XML을 로드하고 공통 전처리 메소드를 호출합니다."""
        try:
            response_json = parse_xml.send_xml_to_api(fpath, api_url)
            if not response_json.get('status'):
                print(f"API 처리 실패: {response_json.get('message')}, 파일: {os.path.basename(fpath)}")
                return None
            
            message, status, metadata, signal = parse_xml.parse_api_response(response_json)
            metadata['statement'] = str(metadata.get('statement', ''))
            
            # 공통 헬퍼 메소드 호출
            return ECGDataset._process_raw_ecg(signal, metadata, fpath, do_bandpass)

        except Exception as e:
            print(f"XML 로딩/파싱 중 예외 발생: {e}, 파일: {os.path.basename(fpath)}")
            return None

    @staticmethod
    def load_ecg_from_pkl(fpath: str, do_bandpass: dict = None) -> Optional[np.ndarray]:
        """Pickle을 로드하고 공통 전처리 메소드를 호출합니다."""
        try:
            with open(fpath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            signal = loaded_data['signal']
            metadata = loaded_data['metadata']

            # 공통 헬퍼 메소드 호출
            return ECGDataset._process_raw_ecg(signal, metadata, fpath, do_bandpass)

        except Exception as e:
            print(f"Pickle 로딩 중 예외 발생: {e}, 파일: {os.path.basename(fpath)}")
            return None


    @staticmethod
    def preprocess_ecg(
        x: np.ndarray,
        target_fs: int = 256,
        do_normalize: bool = False,
        transforms: Optional[Callable] = None,
    ) -> np.ndarray:
        '''
            x: 입력 ECG 신호 (12, target_fs*10)
        '''
        # 1. resample
        if x.shape[-1] != target_fs*10:
            x = resample(x, int(target_fs*10), axis=-1).astype(np.float32)
        # 2. normalize
        if do_normalize:
            clipped_x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
            mu = np.median(x, axis=-1, keepdims=True)
            sigma = np.std(clipped_x, axis=-1, keepdims=True)
            x = (x - mu) / (sigma + 1e-8)
        # 3. transforms
        if transforms:
            x = transforms(x)
        
        return x.astype(np.float32)
    

    # --- 기본 생성자: 이제 파일 경로 리스트를 직접 받음 ---
    def __init__(self,
                 file_paths: List[str],
                 labels: Optional[List[Any]] = None,
                 do_bandpass: Dict = None,
                 do_resample: bool = True,
                 target_fs: int = 256,
                 do_normalize: bool = True,
                 transforms: Optional[Callable] = None,
                 label_transforms: Optional[Callable] = None):
        
        self.file_paths = file_paths
        self.labels = labels
        
        if self.labels is not None:
            assert len(self.file_paths) == len(self.labels), "입력(file_paths)과 레이블(labels)의 개수가 일치해야 합니다."

        self.do_bandpass = do_bandpass
        self.do_resample = do_resample
        self.target_fs = target_fs
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
        _, ext = os.path.splitext(fpath)
        ext = ext.lower()
        if ext == '.pkl':
            x = self.load_ecg_from_pkl(fpath, do_bandpass=self.do_bandpass)
        elif ext == '.xml':
            x = self.load_ecg_from_xml(fpath, do_bandpass=self.do_bandpass, api_url=ECGDataset.XML_API_URL)

        x = self.preprocess_ecg(
            x=x,
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
