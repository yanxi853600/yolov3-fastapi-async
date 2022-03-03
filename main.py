import sys
from urllib3 import Timeout
import uvicorn
from enum import Enum
from typing import Optional
from Inference.errors import Error
from Inference.exceptions import ModelNotFound, InvalidModelConfiguration, ApplicationError, ModelNotLoaded, \
	InferenceEngineNotFound, InvalidInputData
from Inference.response import ApiResponse
from starlette.responses import FileResponse
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, File, UploadFile, Header, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import argparse
from detection import run
from pydantic import BaseModel
from base64 import b64decode, b64encode
import numpy as np
from PIL import Image
import io
import sys
import base64
import cv2
from io import BytesIO
import logging
import logging.config
import os
import time
import datetime
from datetime import date
from detection.models import *
from detection.utils.utils import *
from detection.utils.datasets import *
from detection.utils.augmentations import *
from detection.utils.transforms import *
from detection.return_json import Bbox, Pred, Images
from Inference.errors import Error
from requests.exceptions import Timeout
from traceback import extract_tb
from sys import exc_info,exit
import asyncio
#####################################################
# 	API Release Information (http://127.0.0.1:8888/docs)
#####################################################
app = FastAPI(version="1.0.0", title='Yolov3 inference Swagger',
			  description="<b>API for performing YOLOv3 inference.</b></br></br>"
						  "<b>Contact the developers:</b></br>"
						  "<b>Yanxi.Lin: <a href='mailto:Yanxi.Lin@advantech.com.tw'>Yanxi.Lin@advantech.com.tw</a></b></br>"
			 )
#####################################################
#	CORS Setting
#####################################################
# app.mount("/public", StaticFiles(directory="/main/public"), name="public")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
	max_age=180,	#	timout (second)
)
#####################################################
#	Define Class Object
#####################################################
#yolov3_model = yolov3(args.num_class, args.anchors)
class ModelName(str, Enum):
    # yolov3 = "yolov3"
    R0402="R0402"
    C0402="C0402"
    R0603="R0603"
    C0603="C0603"
    C0201="C0201"
    R0201="R0201"
    C0805="C0805"
    SC705P="SC705P"   
    SOT23="SOT23"
    SOT235P="SOT235P"

#####################################################
#	Loaded yolov3 model
#####################################################

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    parser.add_argument("--model_def", type=str, default="./detection/config/yolov3-SMT_AOI.cfg",
                        help="The path of the anchor txt file.")
    parser.add_argument("--img_size", type=str, default=416,
                        help="The path of the weights to restore.")
    parser.add_argument("--gpu", type=int, default="0",
                        help="number of pad") 
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class load:
	def __init__(self):# , R0402_model, C0402_model, R0603_model, C0603_model
		self.R0402_model_name ="./detection/checkpoints/R0402/R0402_V2_ckpt_245.pth"
		self.C0402_model_name ="./detection/checkpoints/C0402/C0402_V2_ckpt_236.pth"
		self.R0603_model_name ="./detection/checkpoints/R0603/SMT_R0603_20211130.pth"
		self.C0603_model_name ="./detection/checkpoints/C0603/SMT_C0603_20211130.pth"
		self.C0201_model_name ="./detection/checkpoints/C0201/SE_1/SMT_C0201_20220111-C0201-SE-1_399.pth"
		self.R0201_model_name ="./detection/checkpoints/R0201/SE_1/SMT_R0201_20220117_R0201_SE_1_V1_399.pth"
		self.C0805_model_name ="./detection/checkpoints/C0805/SE_1/SMT_C0805_C0805-SE-1_396.pth"
		self.SC705P_model_name ="./detection/checkpoints/SC705P/SE_1/SMT_SC-70-5P_20220106-SE-1_399.pth"
		self.SOT23_model_name ="./detection/checkpoints/SOT23/SE_1/SMT_SOT-23_20220113_SOT-23_SE_1_V2_399.pth"
		self.SOT235P_model_name ="./detection/checkpoints/SOT235P/SE_1/SMT_SOT-23-5P_20220119_SOT_23_5P_SE_1_V1_332.pth"
		

	def model(self):
		# Load pre-trained model
		global R0402_model, C0402_model, R0603_model, C0603_model
		try:
			R0402_model = Darknet(args.model_def, img_size=args.img_size).to(device)
			R0402_model.load_state_dict(torch.load(self.R0402_model_name))
			R0402_model.eval()
			logging.info('--> R0402_model loaded is successfully.')
			C0402_model = Darknet(args.model_def, img_size=args.img_size).to(device)
			C0402_model.load_state_dict(torch.load(self.C0402_model_name)) 
			C0402_model.eval()
			logging.info('--> C0402_model loaded is successfully.')
			R0603_model = Darknet(args.model_def, img_size=args.img_size).to(device)
			R0603_model.load_state_dict(torch.load(self.R0603_model_name))
			R0603_model.eval()
			logging.info('--> R0603_model loaded is successfully.')
			C0603_model = Darknet(args.model_def, img_size=args.img_size).to(device)
			C0603_model.load_state_dict(torch.load(self.C0603_model_name))
			C0603_model.eval()
			logging.info('--> C0603_model loaded is successfully.')
		except Exception as e:
			print('  !!! Model loaded failed : '+ str(e) )
			logging.warning('  !!! Model loaded is failed : '+ str(e) )
		
load_models = load()
load_models.model()
#####################################################
#	Detect an image using yolov3
#####################################################
@app.post('/detect', tags=["POST Method"])
async def predict_image(model_name: ModelName, request: Request, image: UploadFile = File(...)):

	images = await image.read()
	# image = file.read()
	filename = image.filename
	#images = Image.open(io.BytesIO(images))
	# json_data = run.inferenceYoLo(model_name, images, filename).encode('utf-8')
	if model_name == 'R0402':
		model_load =R0402_model
	elif model_name == 'C0402':
		model_load =C0402_model
	elif model_name == 'R0603':
		model_load =R0603_model
	elif model_name == 'C0603':
		model_load =C0603_model

	label = run.inferenceYoLo(model_load, images, filename).encode('utf-8')
	await asyncio.sleep(0.001)
	#cv2.imwrite('./results/'+filename, superimposed_img)
	#res, im_png = cv2.imencode(".jpg", superimposed_img)
		
	#return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")
	return Response(media_type="text/html", content=label)


if __name__ == "__main__":
	try:
		uvicorn.run(app, host="0.0.0.0",port=8888, debug=True)

	except Timeout as ex :
		print("Exception raise: ", ex)
		pass
	except Exception as e:
		error_class = e.__class__.__name__ #取得錯誤類型
		detail = e.args[0] #取得詳細內容
		cl, exc, tb = exc_info() #取得Call Stack
		lastCallStack = extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
		fileName = lastCallStack[0] #取得發生的檔案名稱
		lineNum = lastCallStack[1] #取得發生的行號
		funcName = lastCallStack[2] #取得發生的函數名稱
		errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
		with open('runtime_error.log','a+',encoding='utf-8') as f:
			f.writelines(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\t'+errMsg+'\n')
		print(errMsg)
		exit(0)

