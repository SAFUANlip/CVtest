import copy
import torch
import json
from figures import *
from dataset import Dataset
from ultralytics import YOLO
import os
import wandb
from wandb.yolov8 import add_wandb_callback
from metrics import iou_per_img

# Press the green button in the gutter to run the script.

import warnings


def create_data(save_path, fig_names=["circle", "rhombus", "triangle", "hexagon"]):
    data_train = Dataset(save_path=save_path+"/train", fig_names=fig_names)
    data_train.create(num_of_img=400)

    # data_val = Dataset(save_path=save_path+"/val")
    # data_val.create(num_of_img=2000, fig_names=fig_names)

    # data_test = Dataset(save_path=save_path+"/test", fig_names=fig_names)
    # data_test.create(num_of_img=3000)
    #
    # data_test_with_hexagon = Dataset(save_path=save_path + "/test_with_hexagon", fig_names=["circle", "rhombus", "triangle", "hexagon"])
    # data_test_with_hexagon.create(num_of_img=3000)


def calc_metric(preds_path, labels_path, images_path):
    total_iou = {}
    for f in os.listdir(preds_path):
        name, iou = iou_per_img(preds_path, labels_path, images_path, f[:-4])
        total_iou[name] = iou
    print(np.mean(list(total_iou.values())))

    results = {"img_count": len(total_iou),
               "mean_iou": np.mean(list(total_iou.values())),
               "max_iou": {"img_name": max(total_iou, key=total_iou.get),
                           "iou": total_iou[max(total_iou, key=total_iou.get)]},
               "min_iou": {"img_name": min(total_iou, key=total_iou.get),
                           "iou": total_iou[min(total_iou, key=total_iou.get)]},
            }

    with open(os.getcwd()+"\\"+f"iou_results_final.json", "w") as write_file:
        json.dump(results, write_file)
    write_file.close()

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    return device

def iterative_learning(cycle):
    data_train = Dataset(save_path="C:/Users/Safuan/Python/CVtest/part3/" + "/train", fig_names=["circle", "rhombus", "triangle", "hexagon"])
    device = get_device()

    for i in range(cycle):
        model = YOLO('yolov8n.pt')
        model.train(data='data_part3_test.yaml', project='CVtest_logs_iterative', device=device, epochs=15, batch=256, imgsz=256, flipud=0.4, fliplr=0.4, val=False, save=False)
        model.val(data="data_part3_test.yaml", imgsz=256, conf=0.25, iou=0.7)
        model.val(data="data_part3_hexagon.yaml", imgsz=256, conf=0.25, iou=0.7)
        data_train.create(num_of_img=400)


if __name__ == '__main__':
    # device = get_device()
    # create_data("C:/Users/Safuan/Python/CVtest/part3", fig_names=["circle", "rhombus", "triangle"])

    iterative_learning(20)

    # add_wandb_callback(model, enable_model_checkpointing=True)
    # model.train(data='data.yaml', device=device, project='CVtest_loggs', epochs=40, batch=256, imgsz=256, flipud=0.4, fliplr=0.4)

    #model.val(data="data_test.yaml", imgsz=256, conf=0.25, iou=0.7)

    # calc_metric(preds_path="C:/Users/Safuan/Python/CVtest/runs/detect/predict/labels/",
    #             labels_path="C:/Users/Safuan/Python/CVtest/test/data/labels/",
    #             images_path="C:/Users/Safuan/Python/CVtest/test/data/images/")
