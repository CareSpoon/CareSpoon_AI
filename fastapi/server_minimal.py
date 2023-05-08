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
# from ai_service.yolov5.detect import classification_yolov5
from starlette import status
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import torch
import base64
import io
import json
import os
import time

app = FastAPI()

@app.get("/")
async def home(request: Request):
  ''' Returns barebones HTML form allowing the user to select a file and model '''

  html_content = '''
<form method="post" enctype="multipart/form-data">
  <div>
    <label>Upload Image</label>
    <input name="file" type="file" multiple>
    <div>
      <label>Select YOLO Model</label>
      <select name="model_name">
        <option>yolov5s</option>
        <option>yolov5m</option>
        <option>yolov5l</option>
        <option>yolov5x</option>
      </select>
    </div>
  </div>
  <button type="submit">Submit</button>
</form>
'''

  return HTMLResponse(content=html_content, status_code=200)

@app.post("/image")
async def classification(file: UploadFile = File(...)):
    # model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload = False)
    # model = torch.load('D:\CareSpoon-AI\fastapi\best.pt')
    #This is how you decode + process image with PIL
    # results = model(Image.open(BytesIO(await file.read())))

    #This is how you decode + process image with OpenCV + numpy
    # results = model(cv2.cvtColor(cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR))

    # json_results = results_to_json(results,model)
    # UPLOAD_DIR = '../uploadfile'
    print("----------classification----------")
    try:
        image = await file.read()
        print("image = await file.read()")
        pil_image = Image.open(BytesIO(image))
        print("")
        output = BytesIO()
        pil_image.save(output, format='JPEG')
        content = output.getvalue()
        print("이미지 받기 성공")

        # filename = f"{str(uuid.uuid4())}.jpg"
        # with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        #   fp.write(content)
    except Exception as e:
        # logger.exception(f"create_food_item fail:\n\t{e}\nWrong image")
        # print(e)
        print("성공")
    results = food_classification(content)
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