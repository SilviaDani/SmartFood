#!/usr/bin/env python3
"""
Script di test per l'endpoint CSV upload

Utilizzo:
    python test_csv_upload.py example_data.csv
"""

import requests
import sys
import os

def test_upload(csv_file_path, api_url='http://localhost:8000'):
    """Testa il caricamento del CSV all'API"""
    
    if not os.path.exists(csv_file_path):
        print(f"❌ File not found: {csv_file_path}")
        return False
    
    print(f"📤 Uploading {csv_file_path} to {api_url}/api/csv/upload...")
    
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{api_url}/api/csv/upload', files=files)
        
        print(f"\n📊 Response Status: {response.status_code}")
        data = response.json()
        print(f"📋 Response: {data}")
        
        if response.status_code == 200 and data.get('success'):
            print(f"\n✅ Upload successful! {data.get('rows_processed', 0)} rows processed.")
            return True
        else:
            print(f"\n❌ Upload failed: {data.get('message', 'Unknown error')}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection error: Cannot reach {api_url}")
        print("Make sure the backend is running with: python -m smartfood.csv_uploader")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_csv_upload.py <csv_file> [api_url]")
        print(f"Example: python test_csv_upload.py example_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:8000'
    
    success = test_upload(csv_file, api_url)
    sys.exit(0 if success else 1)
