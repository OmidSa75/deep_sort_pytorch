import os
import cv2
import time
import argparse
import torch
import warnings
import sys
# sys.path.insert(0, 'detector/ScaledYOLOv4')

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_tracking_result)

from detector import build_detector
# from YoloV4Scaled import ScaledYoloV4
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from utils import mmtracking_output, draw_boxes, draw_bbox_keypoints


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        # self.detector = build_detector(cfg, use_cuda=use_cuda)
        # self.detector = ScaledYoloV4()
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.detector.conf = 0.6
        # self.class_names = self.detector.class_names
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # self.det_checkpoint = "http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth"
        # self.det_config_file = '../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
        # self.detector_2 = init_detector(self.det_config_file, self.checkpoint, device="cuda:0")

        self.pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth"
        self.pose_config_file = "../mmpose/configs/top_down/mobilenet_v2/coco/mobilenetv2_coco_384x288.py"
        self.pose_model = init_pose_model(self.pose_config_file, self.pose_checkpoint, device="cuda:0")
        self.dataset = self.pose_model.cfg.data['test']['type']

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.fps = self.vdo.get(cv2.CAP_PROP_FPS)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, self.fps, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.read()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            # im = ori_im.copy()

            '''-----------------------detection part---------------------------'''
            # mmdet_results = inference_detector(self.detector_2, im)
            # bbox_xywh, cls_conf, cls_ids = process_mmdet_results(mmdet_results[0])
            # tic = time.time()
            # bbox_xywh, cls_conf, cls_ids = self.detector(im)
            # tac = time.time()
            # print("\033[1;34m" + str(1 / (tac - tic)))
            pred = self.detector(im).xywh
            pred = pred[0]
            bbox = pred[:, :4]
            bbox_xywh = bbox.cpu().numpy()
            cls_conf = pred[:, 4].cpu().numpy()
            cls_ids = pred[:, 5].long().cpu().numpy()

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            '''------------------------tracking part----------------------------'''
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            '''------------------------pose estimation part--------------------------'''
            # change deep sort tracking result to mmtracking results
            mmtracking_results = mmtracking_output(outputs)
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                im,
                mmtracking_results,
                bbox_thr=0.3,
                format='xyxy',
                dataset=self.dataset,
                return_heatmap=False,
                outputs=None)

            '''------------------------classification part---------------------------'''

            '''------------------------draw part-------------------------'''
            ori_im = draw_bbox_keypoints(ori_im, pose_results)
            # draw boxes for visualization
            # if len(outputs) > 0:
            #     bbox_tlwh = []
            #     bbox_xyxy = outputs[:, :4]
            #     identities = outputs[:, -1]
            #     # ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
            #
            #     for bb_xyxy in bbox_xyxy:
            #         bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
            #
            #     results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
