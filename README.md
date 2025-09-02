# MacAI 심전도(ECG) 분석 추론 엔진

## 1. 프로젝트 개요

본 프로젝트는 심전도(ECG) 데이터를 분석하여 심방세동(Atrial Fibrillation, AF) 및 주요 부정맥(Clinical Important Arrithma, CIA)의  위를 예측하는 FastAPI 기반의 딥러닝 추론 서버입니다.

XML 형식의 ECG 파일을 입력받아 내부적으로 앙상블 모델 추론 및 보정(Calibration)을 수행하고, 최종 위험도 수준(Low, Intermediate, High)과 확률값을 반환합니다.

## 2. 주요 기능

-   **FastAPI 기반 서버**: 비동기 웹 프레임워크를 사용하여 높은 성능을 제공합니다.
-   **XML 데이터 처리**: 외부 XML 파서 API와 연동하여 ECG 데이터를 추출하고 전처리합니다.
-   **앙상블 추론**: 여러 딥러닝 모델(PyTorch)의 결과를 평균 내어 안정적이고 신뢰도 높은 예측을 수행합니다.
-   **확률 보정**: 예측된 확률값을 보정 모델을 통해 실제 발생률에 가깝게 조정합니다.
-   **동적 설정**: `config.yaml` 파일을 통해 모델 경로, 전처리 파라미터 등을 쉽게 변경할 수 있습니다.

## 3. 프로젝트 구조

```
macai-engine-v1/
├── data/                     # 샘플 데이터 (테스트용)
├── notebooks/                # 개발 및 테스트용 Jupyter Notebook
├── service/                  # FastAPI 서비스 애플리케이션
│   ├── main.py               #   - FastAPI 서버 메인 로직
│   └── sbin/
│       └── test_request.sh   #   - 서버 테스트용 쉘 스크립트
├── src/                      # 추론 및 전처리에 사용되는 핵심 소스 코드
│   └── processing/
│       ├── dataset.py
│       ├── infer.py
│       └── transforms.py
├── trained_models/           # 학습된 모델 가중치 및 설정 파일
│   └── v1.3.0/
│       ├── config.yaml       #   - 프로젝트 핵심 설정 파일
│       ├── best_model_1.pth  #   - 모델 가중치 파일 예시
│       └── ...
└── README.md                 # 프로젝트 설명 파일
```


## 4. 설치 및 환경 설정

**사전 요구사항**: Python 3.8 이상, 별도의 XML 파서 서버

1.  **저장소 복제**
    ```bash
    git clone <저장소_URL>
    cd macai-engine-v1
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    conda create -n my_env python=3.10
    conda activate my_env
    ```

3.  **필수 라이브러리 설치**
    프로젝트와 서비스 실행에 필요한 모든 종속성을 함께 설치합니다.
    ```bash
    python -m pip install .[service]
    ```

## 5. 서버 실행 방법

1.  **XML 파서 서버 실행**
    본 추론 엔진은 외부 XML 파서 서버와 통신합니다. **반드시 XML 파서 서버를 먼저 실행해야 합니다.**

2.  **추론 엔진 서버 실행**
    프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.
    ```bash
    uvicorn service.main:app --host 0.0.0.0 --port 8831 --reload
    ```
    -   `--host 0.0.0.0`: 외부에서 서버에 접근할 수 있도록 허용합니다.
    -   `--port 8831`: 8831번 포트를 사용합니다.
    -   `--reload`: 코드 변경 시 서버가 자동으로 재시작됩니다 (개발용).

3.  **테스트 요청 보내기**
    서버가 정상적으로 실행되면, 새 터미널을 열고 테스트 스크립트를 실행하여 서버의 동작을 확인할 수 있습니다.
    ```bash
    cd service/sbin/
    ./test_request.sh
    ```
    스크립트 실행 후, JSON 형식의 위험도 예측 결과가 출력되면 성공입니다.

## 6. API 엔드포인트

### `/predict`

-   **Method**: `POST`
-   **Description**: 하나 이상의 ECG XML 파일을 받아 위험도를 예측합니다.
-   **Request (Form Data)**:
    -   `files`: (`List[UploadFile]`, required) - 하나 이상의 XML 파일.
    -   `xml_parser_url`: (`str`, required) - XML 파서 API의 URL.
    -   `xml_healthcheck_url`: (`str`, required) - XML 파서 상태 확인 URL.
-   **Response (JSON)**:
    ```json
    {
      "af_risk_probability": 0.1234,
      "af_risk_level": "Intermediate",
      "cia_risk_probability": 0.0567,
      "cia_risk_level": "Low"
    }
    ```