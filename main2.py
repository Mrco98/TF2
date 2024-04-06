from predictT import scan_siren
from trafficlight import TL1, TL2, TL3, off, yellow
from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
from time import time
from datetime import datetime
import argparse

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio-Visual Traffic Light System')
    parser.add_argument('--mic_index', type=int, default=2, help='Index of Mic Device')
    args, _ = parser.parse_known_args()

index = args.mic_index

print("********************INITIALIZING PROGRAM********************")
model_path = 'm0dels/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
#device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print("===> LOADING MODEL")
mod3l = YOLO(r'best1.pt')
print("===> MODEL LOADED")
print("===> LOADING CAMERA")
#cap = cv2.VideoCapture('1.mp4')
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("===CAMERA NOT AVAILABLE===")
print("===> CAMERA LOADED")
start_time = time()
frame_number = 0
recheck = 0
listening = True
GPIO.output(8, 0)
print("********************PROGRAM INITIALIZED********************")

def sirenFound(conf):
    print("--------------------EMERGENCY VEHICLE SIREN DETECTED--------------------")
    #print("MADE IT HERE")
    #print(conf.shape)
    #print(type(conf))
    print(f"Confidence: {conf}")
    

def listen():
    #print("<<<<<<<<<<LISTENING FOR EMERGENCY VEHICLE SIREN>>>>>>>>>>")
    #sirenDetected, conf = scan_siren(index)
    sirenDetected = scan_siren(index)
    #print(conf.shape)
    if sirenDetected == True:
        sirenFound(conf)
        #sirenFound()
        #print(sirenDetected)
        #print(conf)
        #print(sirenDetected)
    start_time = time()
    return sirenDetected


while True:
    while listening == True:
        #listen()
        if listen() == False:
            print("<<<<<<<<<<LISTENING FOR EMERGENCY VEHICLE SIREN>>>>>>>>>>")
        else:
        #if listen() == True:
            break

    listening = False
    

    ret, frame = cap.read()
    if not ret:
        break

    print(frame.shape)
    frame = frame * 1.0/255
    frame = torch.from_numpy(np.transpose(frame[:, :, [2, 1, 0]], (2, 0, 1))).float()
    frame_LR = frame.unsqueeze(0)
    frame_LR = frame_LR.to(device)
    print(frame_LR, "-----", frame_LR.shape)
    print(type(frame_LR))
    with torch.no_grad():
        output = model(frame_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    frame = output

    print("FRAME NUMBER =====> ", frame_number)
    frame_number += 1

    if frame_number % 10 != 0: continue

    if recheck > 4:
        listening = True
        GPIO.output(8,0)
        recheck = 0
        continue

    print("SCAN AREA COUNTER =====> ", recheck)
    recheck += 1
    
    results = mod3l(frame, imgsz=640, conf=0.25, verbose=False)
    if len(results[0]) == 0:
        frame_number = 0
        #GPIO.output(8, 0)
        print("<<<<<<<<<<NO EMERGENCY VEHICLE BLINKERS DETECTED>>>>>>>>>>")
        
    for result in results[0]:
        prob = round(float(result.boxes[0].conf), 2)
        if time() - start_time > 1:
            TL3()
            print("|-------------------------------------------------------------------------|")
            print("|------------------|EMERGENCY VEHICLE BLINKER DETECTED|-------------------|")
            print("|CONFIDENCE =====> ", prob, "<================================================|")
            print("|TIME ===========> ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "<================================|")
            print("|-------------------------------------------------------------------------|")
            start_time = time()
            frame_number = 0
            recheck = 0
            GPIO.output(8, 1)

cap.release

