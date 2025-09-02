# 1. 베이스 이미지 선택
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 프로젝트 전체 소스 코드 복사
# pyproject.toml을 사용하기 위해 프로젝트 전체를 복사합니다.
COPY .. /app

# 4. 'service' 의존성 설치
# requirements.txt 대신 pyproject.toml의 'service' 그룹을 직접 설치합니다.
RUN pip install --no-cache-dir .[service]

# 5. Health Check (이전과 동일)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:80/health || exit 1

# 6. 컨테이너 실행 시 FastAPI 서버 구동
# uvicorn 실행 경로를 service 디렉토리 내부로 지정해야 합니다.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "80"]