'''
This file is a barebones FastAPI example that:
  1. Accepts GET request, renders a HTML form at localhost:8000 allowing the user to
     upload a image and select YOLO model, then submit that data via POST
  2. Accept POST request, run YOLO model on input image, return JSON output

Works with client_minimal.py

This script does not require any of the HTML templates in /templates or other code in this repo
and does not involve stuff like Bootstrap, Javascript, JQuery, etc.
'''

from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from yolov5.detect import food_classification, predict_percent
import numpy as np
import cv2, torch, base64, io, json, os, time
from io import BytesIO
from starlette import status
from PIL import Image
from torchvision.transforms import functional as F
import shutil
from pathlib import Path

app = FastAPI()

# model = torch.load("D:/CareSpoon-AI/fastapi/best.pt")
model = torch.load("/home/ubuntu/CareSpoon_AI/fastapi/best.pt")

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # UPLOAD_DIR = 'D:/CareSpoon-AI/fastapi/photo'
    UPLOAD_DIR = '/home/ubuntu/CareSpoon_AI/fastapi/photo'

    if file != None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 디렉토리 생성
        local_path = os.path.normpath(os.path.join(UPLOAD_DIR, file.filename))
        print("local_path")
        print(local_path)
        with open(local_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

    results = food_classification(local_path)

    return results

@app.post("/predict_percent")
def predict_percent(file: UploadFile = File(...)):
    # UPLOAD_DIR = 'D:/CareSpoon-AI/fastapi/photo'
    UPLOAD_DIR = '/home/ubuntu/CareSpoon_AI/fastapi/photo'

    if file != None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 디렉토리 생성
        local_path = os.path.normpath(os.path.join(UPLOAD_DIR, file.filename))
        print("local_path")
        print(local_path)
        with open(local_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

    results = predict_percent(local_path)

    return results

if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server:app'
    uvicorn.run(app_str, host='0.0.0.0', port=8000, reload=True, workers=1)