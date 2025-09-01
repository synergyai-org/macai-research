import os
import requests
import argparse
import numpy as np
import pandas as pd
from glob import glob


def get_xml_filepaths(directory: str) -> list:
    xml_filepaths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith('.xml'):
                xml_filepaths.append(filepath)

    return xml_filepaths


def send_xml_to_api(filepath: str, api_url: str) -> dict:
    filename = os.path.basename(filepath)
    with open(filepath, mode='rb') as f:
        files = {'file': (filename, f, 'application/xml')}
        response = requests.post(api_url, files=files)
        print (response.json().keys())
        if response.status_code == 200:
            return response.json()
        else:
            print (True)
            response.raise_for_status()


def parse_api_response(response: dict) -> tuple:
    message = response['message']
    status  = response['status']
    metadata = response['data']['metadata']
    signal  = response['data']['signal']

    return message, status, metadata, signal


# 시그널 저장
def save_signal_to_npy(xml_filepath: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    signal_filename = os.path.basename(xml_filepath).replace('.xml', '.npy')
    output_path = os.path.join(output_dir, signal_filename)
    
    np.save(output_path, signal)


def save_to_csv(xml_filepath: str, metadata: dict, output_path: str):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    xml_filename = os.path.basename(xml_filepath)
    signal_filename = os.path.basename(xml_filepath).replace('.xml', '.npy')

    
    filename_df = pd.DataFrame([{'xml_filename': xml_filename}])
    signal_filename_df = pd.DataFrame([{'signal_filename': signal_filename}])

    # metadata안에 들어 있는 필드는 다음 필드만 파싱: Xml_type, Rate, Pid, Name, Age, Sex, Date, Time, Statement, Severity, Mdsig
    required_fields = ["xml_type", "version", "scale_factor", "rate", "pid", "name", "age", "sex", "date", "time", "statement", "severity", "mdsig"]
    row = {
        "xml_filename": xml_filename,
        "signal_filename": signal_filename
    }

    for field in required_fields:
        row[field] = metadata.get(field, None)
        if field == "scale_factor":
            xml_type = metadata.get("xml_type", "").lower()
            try:
                row[field] = float(row[field])
            except (ValueError, TypeError):
                if xml_type == "philips":
                    row[field] = 0.005
                elif xml_type == "ge":
                    row[field] = 0.00488
                elif xml_type == "infinitt":
                    row[field] = 0.005
                elif xml_type == "fukudami":
                    row[field] = 0.01
                elif xml_type == "sapphire":
                    row[field] = 0.004880000114440918
                else:
                    row[field] = None
    
    new_df = pd.DataFrame([row])

    if not os.path.isfile(output_path):
        new_df.to_csv(output_path, index=False)
    else:
        existing_df = pd.read_csv(output_path)

        if xml_filename in existing_df["xml_filename"].values:
            print(f"[SKIP] {xml_filename} already exists in {output_path}")
            return
        
        new_df.to_csv(output_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False, default='')
    parser.add_argument('--directory', type=str, required=False, default='')
    parser.add_argument('--signal_dir', type=str, required=False, default='./npy')
    parser.add_argument('--output_path', type=str, required=False, default='./output.csv')
    parser.add_argument('--api_url', type=str, required=False, default='http://localhost:18002/parse/parse_xml')
    args = parser.parse_args()
    
    if args.file:
        xml_filepaths = [args.file]
        for xml_filepath in xml_filepaths:
            response = send_xml_to_api(xml_filepath, args.api_url)
            message, status, metadata, signal = parse_api_response(response)
            # print (status, metadata)
            save_signal_to_npy(xml_filepath, args.signal_dir)
            save_to_csv(xml_filepath, metadata, args.output_path)

    elif args.directory:
        xml_filepaths = get_xml_filepaths(args.directory)
        for xml_filepath in xml_filepaths:
            response = send_xml_to_api(xml_filepath, args.api_url)
            message, status, metadata, signal = parse_api_response(response)
            # print (status, metadata)
            save_signal_to_npy(xml_filepath, args.signal_dir)
            save_to_csv(xml_filepath, metadata, args.output_path)
