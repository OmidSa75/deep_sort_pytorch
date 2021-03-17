import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer,)
from utils.torch_utils import select_device, load_classifier, time_synchronized


class ScaledYoloV4():
    def __init__(self, device='0', weights='./yolov4-p5.pt', imgsz=896):
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def __call__(self, img, augment=False, conf_thresh=0.85, iou_thresh=0.5, classes=0):
        img = letterbox(img, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img, augment=augment)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=classes, agnostic=False)
        if pred[0] is not None:
            pred = pred[0]
            bbox = pred[:, :4]
            bbox_xywh = xyxy2xywh(bbox).cpu().numpy()
            cls_conf = pred[:, 4].cpu().numpy()
            cls_ids = pred[:, 5].long().cpu().numpy()

            return bbox_xywh, cls_conf, cls_ids
        else:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))


if __name__ == '__main__':
    device = select_device('0')
    model = attempt_load('yolov4-p5.pt', map_location=device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    x = torch.randint(0, 255, (1, 3, 896, 896), device=device)
    x = x.float()
    x /= 255.0
    torch_out = model(x)

    torch.onnx.export(
        model,
        x,
        "yolov4-p5.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
