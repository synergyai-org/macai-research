#!/bin/bash

# --- 설정 (이 부분을 직접 수정해서 사용하세요) ---

# 1. 추론 서버 주소
SERVER_URL="http://127.0.0.1:8831/predict"

# 2. XML 파서 API 정보 (config.yaml에서 동일하게 설정된 값 사용)
XML_PARSER_URL="http://localhost:18003/parse/parse_xml"
XML_HEALTHCHECK_URL="http://localhost:18003/healthcheck"
XML_PATHS=(
    "/mnt/home/bgk/macai-engine-v1/data/samples/xmls/10144268_2024-06-02_2024060214062189_2024060214053964.xml"
    "/mnt/home/bgk/macai-engine-v1/data/samples/xmls/10203160_2023-10-04_2023100411441625_2023101110031840.xml"
    "/mnt/home/bgk/macai-engine-v1/data/samples/xmls/10302870_2023-10-17_2023101716092374_2023101716095795.xml"
)

# --- 스크립트 실행 (이 아래는 수정할 필요 없습니다) ---

echo "서버에 요청을 보냅니다: $SERVER_URL"
echo "-------------------------------------"

# curl 명령어에 필요한 -F 옵션을 동적으로 생성
# URL 정보들을 먼저 추가합니다.
curl_args=(
    -F "xml_parser_url=$XML_PARSER_URL"
    -F "xml_healthcheck_url=$XML_HEALTHCHECK_URL"
)

# 파일 경로들을 이어서 추가합니다.
for file_path in "${XML_PATHS[@]}"; do
    if [ -f "$file_path" ]; then
        echo "  -> 파일 추가: $file_path"
        curl_args+=(-F "files=@$file_path")
    else
        echo "  -> 경고: 파일을 찾을 수 없습니다. 건너뜁니다: $file_path"
    fi
done

# 전송할 파일이 하나도 없는 경우 종료
if [ ${#curl_args[@]} -le 2 ]; then # URL만 있고 파일이 없는 경우
    echo "오류: 전송할 유효한 파일이 없습니다."
    exit 1
fi

echo "-------------------------------------"
echo "요청 실행..."

# curl 명령어 실행
curl -X POST "$SERVER_URL" "${curl_args[@]}"

echo ""
echo "-------------------------------------"
echo "요청 완료."