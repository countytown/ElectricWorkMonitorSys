import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import filterbox
import numpy as np


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def detect(save_img=True):

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # if half:
    model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once



    A_fps = 0.0
    fps = 0.0
    count = 0
    for path, img, im0s, vid_cap in dataset:
        # vid_writer = None
        # if dataset.mode != 'images':
        #     fourcc = 'mp4v'  # output video codec
            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # save_path = './inference/result.mp4'
            # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))


        t00 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]


        # opt.conf_thres = 0.4

        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=True)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # cv2.imshow('deep', im0)
            # cv2.waitKey(1)

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 6].unique():
                    n = (det[:, 6] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                fps = (fps + (1. / (time.time() - t00))) / 2
                count += 1


                bboxs = []
                cls_confs = []
                cls_idss = []
                # xywh = wywha 有5个值
                bboxs = []
                cls_confs = []
                cls_idss = []
                for i in range(len(pred)):
                    # print(pred[0],'ppppp')
                    bbox = torch.FloatTensor([]).reshape([0, 5])
                    cls_conf = torch.FloatTensor([])
                    cls_ids = torch.LongTensor([])
                    if isinstance(pred[i], torch.Tensor):
                        bbox = pred[i][:, :5]
                        cls_conf = pred[i][:, 5]
                        cls_ids = pred[i][:, 6].long()
                    bboxs.append(bbox.cpu().numpy())
                    cls_confs.append(cls_conf.cpu().numpy())
                    cls_idss.append(cls_ids.cpu().numpy())
                # cv2.imshow('v',im0)
                # cv2.waitKey(1)
                # print(bbox)
                processed_img1 = filterbox.process_height_result(im0, bboxs, cls_confs, cls_idss,path,opt,vid_writer=None)
                processed_img = copy.deepcopy(processed_img1)
                A_fps += fps
                avgfps = A_fps /count
                # print(fps) = im
                print('111')

            else:
                processed_img = copy.deepcopy(im0)
                #processed_img = im0
                print('222')

            # Save results (image with detections)
            save_path = str(Path(out) / Path(p).name)
            if save_img:
                if dataset.mode == 'images':
                    # print(save_path, 'xxxxxxxxxx')
                    cv2.imwrite(save_path, processed_img1)

                    # cv2.imshow('xxx',processed_img)
                    # cv2.waitKey(1)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:
                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        # cv2.imshow('home',processed_img)
                        # cv2.waitKey(1)
                        vdo_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vdo_writer.write(processed_img)


            # cv2.imshow('deep', processed_img)
            # # cv2.imshow('deep1', im0)
            # cv2.waitKey(10)


                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, processed_img)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # # Save results (image with detections)
                # save_path = str(Path(out) / Path(p).name)
                # if save_img:
                #     if dataset.mode == 'images':
                #         # print(save_path, 'xxxxxxxxxx')
                #         cv2.imwrite(save_path, processed_img1)
                #
                #         # cv2.imshow('xxx',processed_img)
                #         # cv2.waitKey(1)
                #     else:
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer
                #             if vid_cap:
                #                 fourcc = 'mp4v'  # output video codec
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #
                #             else:
                #                 fps,w,h = 30,im0.shape[1],im0.shape[0]
                #                 save_path += '.mp4'
                #             # cv2.imshow('home',processed_img)
                #             # cv2.waitKey(1)
                #             vdo_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vdo_writer.write(processed_img)

    #
    # # if save_txt or save_img:
    # #     print('Results saved to %s' % Path(out))
    # #
    # # print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/m-70.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output1', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=720, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.06, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--ori', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--simulate', nargs='?', const=True, default=False, help='resume most recent training')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()