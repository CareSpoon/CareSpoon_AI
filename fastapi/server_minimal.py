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
from yolov5.detect import food_classification
import numpy as np
import cv2, torch, base64, io, json, os, time
from io import BytesIO
from starlette import status
from PIL import Image
from torchvision.transforms import functional as F
import shutil

app = FastAPI()

model = torch.load("D:/CareSpoon-AI/fastapi/best.pt")

def preprocess_image(image):
    image = F.resize(image, (640, 640))  # 이미지 크기 조정 (적절한 크기로 수정 가능)
    image = F.to_tensor(image)  # 텐서로 변환
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    return image.unsqueeze(0)
    # return image

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 업로드된 이미지를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    try:
        predictions = food_classification(image)
    except Exception as e:
        print(e)

    return predictions  # 예시로 추론 결과를 딕셔너리 형태로 반환
    # return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 업로드된 이미지를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    # 이미지 전처리
    input_tensor = preprocess_image(image)
    
    # 모델 추론
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # 추론 결과 반환
    return predictions

@app.post("/file")
def file(file: UploadFile = File(...)):
    UPLOAD_DIR = 'fastapi\photo'
    SERVER_IMG_DIR = os.path.normpath(os.path.join('http://localhost:8000/', 'file/'))

    if file != None:
        local_path = os.path.normpath(os.path.join(UPLOAD_DIR, file.filename))
        print(SERVER_IMG_DIR)
        print(local_path)
        with open(local_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        server_path = os.path.join(SERVER_IMG_DIR, file.filename)

    return {
        "content_type": file.content_type,
        "filename": file.filename,
        "server_path": server_path
    }


@app.post("/photo")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./photo"  # 이미지를 저장할 서버 경로
    
    # if not os.path.exists(UPLOAD_DIR):
    #     os.makedirs(UPLOAD_DIR)

    content = await file.read()
    
    filename = f"{file.filename}.jpg"  # uuid로 유니크한 파일명으로 변경
    filepath = os.path.join(UPLOAD_DIR, filename)

    print("filepath")
    print(os.path.normpath(filepath))

    with open(os.path.normpath(filepath), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    return {"filename": filename}

@app.post("/classification2")
async def classification2(file: UploadFile = File(...)):
    # 파일 저장 경로 및 파일명 설정
    UPLOAD_DIR = "./temp"
    filename = f"{file.filename}.jpg"
    file_path = f"./temp/{file.filename}"
    
    img = await file.read()
    
    print(os.path.join(UPLOAD_DIR, filename))
    # 이미지 파일을 읽고 저장
    # with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(img)
    
    # 저장된 이미지 정보 반환
    print("file_name")
    print(filename)
    return {"filename": filename}
    # results = []
    # try:
    #     img = Image.open(BytesIO(await file.read()))
    #     img.save('./temp/tmp.jpg', format='JPEG')

    #     img = Image.open('./temp/tmp.jpg')
    #     results = food_classification(img)
    # except Exception as e:
    #     print("이미지 받기 실패")
    #     print(e)
    
    # print(results)
    # return results

@app.post("/classification")
async def classification(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        output = io.BytesIO()
        img.save(output, format='JPEG')
        byte_string = output.getvalue()
    except Exception as e:
        print("이미지 받기 실패")
    results = food_classification(byte_string)
    print(results)
    return results

def results_to_json(results, model):
    ''' Helper function for process_home_form()'''
    return [
        [
          {
          "class": int(pred[5]),
          "class_name": model.model.names[int(pred[5])],
          # "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
          # "confidence": float(pred[4]),
          }
        for pred in result
        ]
      for result in results.xyxy
      ]


if __name__ == '__main__':
    import uvicorn
    
    app_str = 'server_minimal:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)