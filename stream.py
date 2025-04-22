import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import numpy as np
import cv2
import subprocess
from dataset.annotate import draw, get_dart_scores
import pickle
from predict import bboxes_to_xy

def predict_stream(yolo):
    while True:
        # Capture an image using libcamera-still at 800x800 resolution
        # subprocess.run(["libcamera-still", "-o", "frame.jpg", "--nopreview", "-t", "1", "--width", "800", "--height", "800"])
        # Call the Python 3.11 script to capture an image
        subprocess.run(["/home/pi/Desktop/Automatic-Darts-Scoring/Server/myenv/bin/python3", "capture.py"])  # Path to Python 3.11
        
        # Load the image using OpenCV
        frame = cv2.imread("frame.jpg")
        if frame is None:
            print("Failed to load image")
            continue
        
        # crop image
        # frame = frame[60:685, 50:760]
        # resize: 800x800
        # frame = cv2.resize(frame, (800, 800))
        
        # Flip the frame 180 degrees since camera is upside down
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolo.predict(frame)
        preds = bboxes_to_xy(bboxes, 3)
        xy = preds
        xy = xy[xy[:, -1] == 1]
        frame = draw(frame, xy[:, :2], cfg, circles=False, score=True)
        
        cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == ord('z'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_utrecht')
    args = parser.parse_args()
    
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg
    
    yolo = build_model(cfg)
    yolo.load_weights(osp.join('old-models', args.cfg, 'weights'), cfg.model.weights_type)
    
    predict_stream(yolo)
