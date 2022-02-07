import argparse
from ast import Break
from lzma import PRESET_DEFAULT
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle
import threading

class VideoStreamWidget(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.frames_list = []
        if self.capture.isOpened()==False :
            print("Error running video")
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                self.frames_list.append(self.frame)

def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print('Missed more than 1 calibration point')
    return xy


def predict(
        yolo,
        cfg,
        labels_path='./dataset/labels.pkl',
        dataset='d1',
        split='val',
        max_darts=3,
        write=False):

    np.random.seed(0)

    ############## CODE FOR IMAGES ##################
    # source='/home/linuxuser/Anas/Personal Projects/Deep Darts/cropped_images_dart/cropped_images/800/d1_02_13_2020'
    # img_paths=os.listdir(source)

    # for i, p in enumerate( img_paths):
    #     if i == 1:
    #         ti = time()

    #     path= source + '/' + p
    #     img = cv2.imread(path)
    #     img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     print(np.shape(img))
    #     bboxes = yolo.predict(img_cvt)
    #     prediction = bboxes_to_xy(bboxes, max_darts)
    #     xy = prediction
    #     xy = xy[xy[:, -1] == 1]
        
    #     img = draw(cv2.cvtColor(img_cvt, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
    #     scores=sum(get_dart_scores(prediction[:, :2], cfg, numeric=True))    
    #     print('Score:', scores)
    #     cv2.putText(img, str(scores), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
    #     cv2.imshow('Result', img)
    #     cv2.waitKey()


    ######### CODE FOR VIDEO ##########    
    video_path='/home/linuxuser/Anas/Personal Projects/Deep Darts/dart-videos/dart-video-01.mp4'
    # vid_widget=VideoStreamWidget(video_path)
    
    # while (vid_widget.capture.isOpened()):
    #     try: 
    #         img = vid_widget.frames_list.pop(0)
    #         #img = vid_widget.frame
    #         img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         bboxes = yolo.predict(img_cvt)
    #         prediction = bboxes_to_xy(bboxes, max_darts)
    #         xy = prediction
    #         xy = xy[xy[:, -1] == 1]
            
    #         img = draw(cv2.cvtColor(img_cvt, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
    #         scores=sum(get_dart_scores(prediction[:, :2], cfg, numeric=True))    
    #         print('Score:', scores)
    #         cv2.putText(img, str(scores), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
    #         cv2.imshow('Result', img)
    #         cv2.waitKey(1)
        
    #     except IndexError:
    #         pass


    ##### SECOND CODE FOR VIDEO #######
    capture = cv2.VideoCapture(video_path)
    result = cv2.VideoWriter('result.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (640, 352))
    while(capture.isOpened()):
        ret, img = capture.read()
        if ret == True:
                img=img[140:360, 72:286]
                img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # shape=np.shape(img_cvt)
                # size=(shape[0], shape[1])
                bboxes = yolo.predict(img_cvt)
                prediction = bboxes_to_xy(bboxes, max_darts)
                xy = prediction
                xy = xy[xy[:, -1] == 1]
                
                img = draw(cv2.cvtColor(img_cvt, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
                scores=sum(get_dart_scores(prediction[:, :2], cfg, numeric=True))    
                # print('Score:', scores)
                cv2.putText(img, str(scores), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
                # result.write(img)
                cv2.imshow('Result',img)
                cv2.waitKey(1)
        else:
            break

    capture.release()
    cv2.destroyAllWindows
 


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d1')
    parser.add_argument('-s', '--split', default='val')
    parser.add_argument('-w', '--write', action='store_true')
    parser.add_argument('-f', '--fail-cases', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict(yolo, cfg,
            dataset=cfg.data.dataset,
            split=args.split,
            write=args.write)