import copy
import json
import torch
from figures import *
import matplotlib.pyplot as plt
from dataset import Dataset
from ultralytics import YOLO

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    # data = Dataset(save_path="D:/Work/CVtest")
    # data.create(num_of_img=5000, fig_names=["circle", "rhombus", "triangle", "hexagon"])
