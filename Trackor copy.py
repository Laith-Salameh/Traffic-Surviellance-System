import cv2

from numpy import random
import numpy as np
import time

from Detector import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


from utils.datasets import LoadImages

from collections import deque
import random
import torch




class Trackor:
    def __init__(self , names ,source,confidence,IOU_threshold,assigned_class_id ):
        self.trail_deque_dict = {}
        self.names = names
        self.source = source
        self.imagesize = 448
        self.trail_max_length = 64
        #initialize detector
        self.detector = Detector(names,confidence,IOU_threshold, assigned_class_id,'weights/last_002_e.pt',self.imagesize,'cfg/yolor_csp.cfg')

        # initialize deepsort
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

       



    def _save_restults(self,frame, bbox, identities):
        txt_path = 'res.txt'
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            id = identities[i]
            save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
            with open(txt_path , 'a') as f:
                    line = save_format.format(frame=frame, id=id, x1=int(box[0]), y1=int(box[1]), w=int(box[2]- box[0]), h=int(box[3]-box[1]), x = -1, y = -1, z = -1)
                    f.write(line)

    def Track_and_estimate_Speed(self):    
        dataset = LoadImages(self.source, img_size=self.imagesize, auto_size=64)
        prevTime = 0
        for path, img, im0, vid_cap , frame in dataset:
            #detect
            xywhs, confss, oids = self.detector.detect(img,im0)
            #update Trackor
            
            if xywhs is not None:
                outputs = self.deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    print('***',frame , '***')
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    #save_results
                    self._save_restults(frame ,  bbox_xyxy, identities)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            print('{:.1f}'.format(fps),'\n')

def main():
    names = ['other' , 'car' , 'van' , 'bus']
    assigned = [1 , 3]
    trackor = Trackor(names,'content/dataa/',0.1,0.65,assigned)
    with torch.no_grad():
            trackor.Track_and_estimate_Speed()

if __name__ == "__main__":
    main()