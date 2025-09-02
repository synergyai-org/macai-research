import os
import sys
import yaml
import joblib
import numpy as np
import torch
import warnings
import tempfile
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import requests

# --- 초기 설정 ---
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processing import infer, transforms, dataset

# --- 추론 서비스 클래스 ---
class InferenceService:
    """설정 로드, 모델 캐싱, 추론 로직을 캡슐화하는 클래스"""
    _instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self.config['MODEL_INFO'].get('DEVICE', 'cpu')
        
        print("모델 로드를 시작합니다...")
        self.dl_models = infer.load_models(
            self.config['MODEL_INFO']['MODEL_PATHS'],
            self.config['MODEL_INFO']['CONFIG'],
            device=self.device
        )
        self.af_cal_model = joblib.load(self.config['MODEL_INFO']['AF_CAL_MODEL_PATH'])
        self.cia_cal_model = joblib.load(self.config['MODEL_INFO']['CIA_CAL_MODEL_PATH'])
        print(f"모델 {len(self.dl_models)}개 로드 완료. Device: {self.device}")

        print("전처리 파이프라인을 생성합니다...")
        pipeline_steps = [
            getattr(transforms, info['type'].split('(')[0])(**{k: v for k, v in info.items() if k != 'type'})
            for info in self.config['PREPROCESS']['transforms']['val']
        ]
        self.transform_pipeline = transforms.TransformPipeline(pipeline_steps)
        print("전처리 파이프라인 생성 완료.")
        InferenceService._instance = self

    @classmethod
    def get_instance(cls) -> 'InferenceService':
        if cls._instance is None:
            raise RuntimeError("서비스가 초기화되지 않았습니다.")
        return cls._instance

    def _preprocess_batch(self, ecg_arrays: List[np.ndarray]) -> torch.Tensor:
        """ECG 배열 리스트를 전처리하고 하나의 배치 텐서로 만듭니다."""
        batch_ecg_tensor = []
        for ecg_array in ecg_arrays:
            preprocessed_ecg = dataset.ECGDataset.preprocess_ecg(
                x=ecg_array,
                target_fs=self.config['PREPROCESS']['target_fs'],
                do_normalize=self.config['PREPROCESS']['do_normalize'],
                transforms=self.transform_pipeline
            )
            batch_ecg_tensor.append(preprocessed_ecg)
        return torch.tensor(np.stack(batch_ecg_tensor), dtype=torch.float32).to(self.device)

    def _run_inference(self, ecg_tensor: torch.Tensor) -> np.ndarray:
        """앙상블 추론을 수행하고 평균 확률을 계산합니다."""
        results = [infer.inference_tensor(m['model'], ecg_tensor) for m in self.dl_models]
        probs_list = [result['probs'].cpu() for result in results]
        stacked_probs = torch.stack(probs_list, dim=0)
        return stacked_probs.mean(dim=0).numpy()

    def _classify(self, mean_probs: np.ndarray) -> Tuple[List, List]:
        """보정 및 위험도 분류를 수행합니다."""
        cls_info = self.config['MODEL_INFO']['OUTPUT_CLS_INFO']
        af_idx = cls_info['cls_names'].index(cls_info['target_cls']['AF'])
        cia_idx = cls_info['cls_names'].index(cls_info['target_cls']['CIA'])

        af_prob_cal = self.af_cal_model.predict_proba(mean_probs[:, af_idx].reshape(-1, 1))[:, 1]
        cia_prob_cal = self.cia_cal_model.predict_proba(mean_probs[:, cia_idx].reshape(-1, 1))[:, 1]

        risk_thr = self.config['MODEL_INFO']['RISK_THRESHOLD']
        af_risk = infer.classify_risk(af_prob_cal, risk_thr['af_high_g_thr'], risk_thr['af_inter_g_thr'], method='mean')
        cia_risk = infer.classify_risk(cia_prob_cal, risk_thr['cia_high_g_thr'], risk_thr['cia_inter_g_thr'], method='mean')
        
        return af_risk, cia_risk

    def predict_from_files(self, files: List[UploadFile], xml_parser_url: str, xml_healthcheck_url: str) -> Tuple[List, List]:
        """여러 파일을 하나의 배치로 묶어 전체 추론 파이프라인을 실행합니다."""

        # 1. healthcheck 먼저 수행
        try:
            resp = requests.get(xml_healthcheck_url, timeout=5)
            resp.raise_for_status()
            health = resp.json()
            if not (health.get("success") and health.get("message") == "OK"):
                raise RuntimeError(f"XML 파서 healthcheck 실패: {health}")
        except Exception as e:
            raise RuntimeError(f"XML 파서 healthcheck 오류: {e}")

        # 2. 파일 파싱 진행
        ecg_arrays = []
        for file in files:
            temp_dir = tempfile.mkdtemp()
            try:
                temp_file_path = os.path.join(temp_dir, file.filename)
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                ecg_array = dataset.ECGDataset.load_ecg_from_xml(
                    fpath=temp_file_path,
                    do_bandpass=self.config['PREPROCESS']['do_bandpass'],
                    api_url=xml_parser_url  # 전달받은 URL 사용
                )
                if ecg_array is not None:
                    ecg_arrays.append(ecg_array)
                else:
                    print(f"Warning: Failed to parse ECG from {file.filename}")
            finally:
                shutil.rmtree(temp_dir)

        if not ecg_arrays:
            raise ValueError("유효한 ECG 데이터를 가진 XML 파일이 없습니다.")
        
        batch_tensor = self._preprocess_batch(ecg_arrays)
        mean_probs = self._run_inference(batch_tensor)
        af_risk, cia_risk = self._classify(mean_probs)
        
        return af_risk, cia_risk

# --- FastAPI 설정 ---
service: InferenceService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("서버 시작: 추론 서비스를 초기화합니다.")
    try:
        config_path = os.getenv("CONFIG_PATH", os.path.join(project_root, "trained_models/v1.3.0/config.yaml"))
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        service = InferenceService(config)
        print("추론 서비스가 성공적으로 초기화되었습니다.")
    except Exception as e:
        raise RuntimeError(f"치명적 오류: 서비스 초기화 실패 - {e}")
    
    yield
    
    print("서버 종료.")

app = FastAPI(lifespan=lifespan)

# --- API 입출력 모델 ---
class RiskOut(BaseModel):
    af_risk_probability: float
    af_risk_level: str
    cia_risk_probability: float
    cia_risk_level: str

# --- API 엔드포인트 ---
@app.get("/")
def read_root():
    return {"message": "MacAI Inference Engine v1.3.0 is running"}


@app.post("/predict", response_model=RiskOut)
async def predict_endpoint(
    files: List[UploadFile] = File(...),
    xml_parser_url: str = Form(...),
    xml_healthcheck_url: str = Form(...),
):
    try:
        service = InferenceService.get_instance()
        af_risk, cia_risk = service.predict_from_files(files, xml_parser_url, xml_healthcheck_url)  # <--- url도 넘김
        
        return RiskOut(
            af_risk_probability=af_risk[0],
            af_risk_level=af_risk[1],
            cia_risk_probability=cia_risk[0],
            cia_risk_level=cia_risk[1]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {e}")