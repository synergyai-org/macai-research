import os
import requests
import argparse
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm

def send_xml_to_api(filepath: str, api_url: str) -> dict:
    filename = os.path.basename(filepath)
    with open(filepath, mode='rb') as f:
        files = {'file': (filename, f, 'application/xml')}
        response = requests.post(api_url, files=files)
        # print (response.json().keys())
        if response.status_code == 200:
            return response.json()
        else:
            # print (True)
            response.raise_for_status()

def parse_api_response(response: dict) -> tuple:
    message = response['message']
    status  = response['status']
    metadata = response['data']['metadata']
    signal  = response['data']['signal']
    return message, status, metadata, signal


def process_xml_batch(
    xml_filepaths: List[str], 
    api_url: str
) -> List[Dict[str, Any]]:
    """
    주어진 XML 파일 경로 리스트를 배치 처리하여 각 파일에 대한 API 응답 리스트를 반환합니다.
    파일별 예외처리를 포함하여, 하나의 파일 실패가 전체 배치 작업을 중단시키지 않습니다.

    Args:
        xml_filepaths (List[str]): 처리할 XML 파일의 전체 경로 리스트.
        api_url (str): XML 파싱을 위한 API의 URL.

    Returns:
        List[Dict[str, Any]]:
        - 각 XML 파일에 대한 API 응답(JSON 딕셔너리)의 리스트.
        - 특정 파일 처리 중 오류 발생 시, 해당 요소는 오류 정보를 담은 딕셔너리가 됩니다.
    """
    results = []
    
    for xml_filepath in tqdm(xml_filepaths, desc="Processing XML files"):
        try:
            # API로 XML 파일 전송 및 응답 받기
            response_json = send_xml_to_api(xml_filepath, api_url)
            # 성공한 응답에 파일 경로 추가하여 어떤 파일의 결과인지 명시
            response_json['filepath'] = xml_filepath 
            results.append(response_json)

        except Exception as e:
            # requests 실패, JSON 파싱 실패 등 예상치 못한 에러 처리
            error_info = {
                'status': 'error',
                'message': f"파일 처리 중 예외 발생: {e}",
                'filepath': xml_filepath,
                'data': None
            }
            print(f"파일 처리 실패: {os.path.basename(xml_filepath)}, 이유: {e}")
            results.append(error_info)
            
    return results