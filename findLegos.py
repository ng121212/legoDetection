from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import glob
import math
from collections import deque
#from scipy import ndimage  # might not have this installed!!
import imutils
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
import serial
import serial.tools.list_ports  # for listing serial ports
import random
from enum import Enum
#import pygame
#from pygame.locals import *
import time
import datetime
import argparse
from threading import Thread

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5

# for roaming when no legos in sight
currentRoamInstr = 0


def com_connect():
    ser = None;
    connection_made = False
    while not connection_made:
        if os.path.exists('/dev/ttyUSB0'):
            connection_made = True
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        if os.path.exists('/dev/ttyACM1'):
            connection_made = True
            ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
    return ser

def send_serial_command(serialPort, direction_enum):
    # If this command is different than the last command sent, then we should sent it
    # Or if it's the same command but it's been 1 second since we last sent a command, then we should send it
    if serialPort is not None:
        global lastCommandSentViaSerial
        global lastCommandSentViaSerialTime

        if direction_enum == Direction.RIGHT:
            dataToSend = b'r\n'
        elif direction_enum == Direction.LEFT:
            dataToSend = b'l\n'
        elif direction_enum == Direction.FORWARD:
            dataToSend = b'f\n'
        else:
            dataToSend = b'h\n'

        if lastCommandSentViaSerial != direction_enum:
            serialPort.write(dataToSend)
            time.sleep(1.5)
            lastCommandSentViaSerial = direction_enum
            lastCommandSentViaSerialTime = time.time()
        # elif (time.time() - lastCommandSentViaSerialTime > 1):
        #    serialPort.write(dataToSend)
        #    lastCommandSentViaSerialTime = time.time()
        else:
            pass  # Do nothing - same command sent recently

os.mkdir("output")
os.mkdir("runImages")
os.makedirs("data/samples")
        
# Variables to hold last command sent to Arudino and when it was sent (epoch seconds)
lastCommandSentViaSerial = None
lastCommandSentViaSerialTime = None
prevDirection = None

# Attempt connection to Serial Port to Arduino ... com_connect() hangs until connection is made
serialPort = None
serialPort = com_connect()

# Connect to webcam video source and allow camera to warm up......
vs = WebcamVideoStream(src=1)
vs.start()
time.sleep(2.0)
time.sleep(5.0)
# TODO delete this block when done
start = time.time()
num_frames = 0

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_final.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

classes = load_classes(opt.class_path)  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

count = 1
log = open('runLog.txt', 'a')

# would be infinite (200 just for testing purposes)
while count<200:

    if count%5 == 0:
        send_serial_command(serialPort, Direction.STOP)
        prevDirection = Direction.STOP
        #grab the current frame, put it into the image folder
        frame = vs.read()
        if frame is None:
            print("No frame")
            break
        frame = imutils.resize(frame, width=416)
        cv2.imwrite("runImages/frame%d.jpg" % count, frame)

    elif count == 1 or prevDirection == Direction.STOP:
        #grab the current frame, put it into the image folder
        frame = vs.read()
        if frame is None:
            print("No frame")
            break

        # Resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=416)
        # save image
        cv2.imwrite("data/samples/frame.jpg", frame)
        cv2.imwrite("runImages/frame%d.jpg" % count, frame)


        #execute this once per image
        dataloader = DataLoader(
            ImageFolder(opt.image_folder, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        x = []
        #only runs once really, only processing one image at a time
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            img = np.array(Image.open(path))

            if detections is not None:
            # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

                # x2 always comes after x1, out of 416(?)
                # there will probably be multiple detections per image, need to first store all of them
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x.append((x1+x2)/2)

        # delete image from folder
        os.remove("data/samples/frame.jpg")

        found = False;
        #check if any legos are in the center
        for i in range(len(x)):
            if x[i] >= 175 and x[i] <= 225:
                send_serial_command(serialPort, Direction.FORWARD)
                prevDirection = Direction.FORWARD
                found = True
                break
        if found == False:
            # check if any legos are in the right
            for i in range(len(x)):
                if x[i] > 225:
                    send_serial_command(serialPort, Direction.RIGHT)
                    prevDirection = Direction.RIGHT
                    found = True
                    break
        if found == False:
            # check if any legos are in the left
            for i in range(len(x)):
                if x[i] < 175:
                    send_serial_command(serialPort, Direction.LEFT)
                    prevDirection = Direction.LEFT
                    found = True
                    break
        # no legos in view, so roam
        # (should travel in a square more or less)
        # need to implement degrees of turn on arudino for better performance
        if found == False:
            log.write("DID NOT find lego in frame below")
            print("DID NOT find lego in frame below")
            if currentRoamInstr < 3:
                send_serial_command(serialPort, Direction.RIGHT)
                prevDirection = Direction.RIGHT
            elif currentRoamInstr < 6:
                send_serial_command(serialPort, Direction.LEFT)
                prevDirection = Direction.LEFT
            elif currentRoamInstr < 12:
                send_serial_command(serialPort, Direction.FORWARD)
                prevDirection = Direction.FORWARD         
            currentRoamInstr = (currentRoamInstr+1)%12
        else:
            log.write("found lego in frame ")
            print("found lego in frame ")
    else:
        #grab the current frame, put it into the image folder
        frame = vs.read()
        if frame is None:
            print("No frame")
            break
        frame = imutils.resize(frame, width=416)
        cv2.imwrite("runImages/frame%d.jpg" % count, frame)

    log.write("%d\n" % (count))
    log.write("direction for frame above: ")
    if prevDirection == Direction.LEFT:
        log.write("LEFT\n")
    elif prevDirection == Direction.RIGHT:
        log.write("RIGHT\n")
    elif prevDirection == Direction.FORWARD:
        log.write("FORWARD\n")
    else:
        log.write("STOP\n")

    print(count)
    print("direction after current frame:")
    print(prevDirection)
    count = count+1
    lastCommandSentViaSerial = prevDirection

send_serial_command(serialPort, Direction.STOP)
log.close()
vs.stop()
