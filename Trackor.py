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


from Cookie import *
import pydeck as pdk
import pandas as pd

class Trackor:
    def __init__(self , names ,source,confidence,IOU_threshold,assigned_class_id ,tform , map_image):
        self.trail_deque_dict = {}
        self.tform = tform
        self.names = names
        self.source = source
        self.imagesize = 640
        self.trail_max_length = 64
        self.map = map_image
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

        self.dict = {'lat': [], 'lon': []}
        

       



    def _compute_color_for_labels(self,label):
        if label == 1:  # car  #BGR
            color = (85, 45, 255)
        elif label == 2:  # van
            color = (222, 82, 175)
        elif label == 3:  # bus
            color = (0, 204, 255)
        elif label == 0:  # other
            color = (0, 149, 255)
        else:
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            color = (r,g,b)
        return tuple(color)



    def _UI_box(self,x, img, label=None, color=None, line_thickness=None , speed=0):
        # Plots label and speed over vehicle
        c1 = (int(x[0]), int(x[1]))
        if label:
            tf = max(line_thickness - 1, 1)  # font thickness
            cv2.putText(img, ("{},{}").format(label,speed), (c1[0], c1[1] - 2), 0, fontScale = line_thickness / 3, color = [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    def _estimate_Speed(self,img, bbox, object_id, identities):
        
        for i, box in enumerate(bbox):
            #get class name for object class id
            label1 = '%s' % (self.names[object_id[i]])
            id = int(identities[i]) if identities is not None else 0
            color = self._compute_color_for_labels(object_id[i])

            n = len(self.trail_deque_dict[id])
            cookie = Cookie()
            scale_w = cookie.get('scale_w')
            scale_h = cookie.get('scale_h')
            ppm = cookie.get('ppm')
            if n>1:
                p1 = self.tform.inverse((self.trail_deque_dict[id][1][0], self.trail_deque_dict[id][1][1] ))
                p2 = self.tform.inverse((self.trail_deque_dict[id][0 ][0] , self.trail_deque_dict[id][0 ][1]  ))
                x1 , y1 = (p1[0][0] , p1[0][1])
                x2 , y2 = (p2[0][0] , p2[0][1])
                fps = cookie.get('fps')
                speed = math.sqrt( ((x2 - x1)*scale_w)**2 + ((y2 - y1)*scale_h)**2) * fps * ppm
                self._UI_box(box, img, label=label1, color=color, line_thickness=3 , speed = int(speed))
            else:
                self._UI_box(box, img, label=label1, color=color, line_thickness=3)

    def _drawonmap(self,img, bbox, object_id, identities=None):
        cookie = Cookie()
        self.dict = {'lat': [], 'lon': []}
        la1 , lo1 ,la2 , lo2  = 100000*np.float64(cookie.get("coords"))
        scalex = abs(lo2 - lo1)
        scaley = abs(la2 - la1)
        lo = min(lo1,lo2)
        la = max(la1,la2)
        for i, box in enumerate(bbox):
            #get class name for object class ids
            id = int(identities[i]) if identities is not None else 0
            color = self._compute_color_for_labels(object_id[i])
            n = len(self.trail_deque_dict[id])
            if n>=1:
                p1= self.tform.inverse((self.trail_deque_dict[id][0 ][0] , self.trail_deque_dict[id][0 ][1]  ))
                x1 , y1 = (np.float64(p1[0][0]) , np.float64(p1[0][1]))
                x1 = lo  + (x1* scalex)/640 
                y1 = la  - (y1 * scaley)/512
                self.dict['lon'].append(x1/100000)
                self.dict['lat'].append(y1/100000)

    def _generateHeat(self,img , res, bbox, identities):
        h,w = res.shape[0] , res.shape[1]
        zeros = np.zeros((h,w),np.uint8)
        for i, box in enumerate(bbox):
            id = int(identities[i]) if identities is not None else 0
            n = len(self.trail_deque_dict[id])
            if n>1:
                p1 = self.tform.inverse((self.trail_deque_dict[id][1][0], self.trail_deque_dict[id][1][1] ))
                p2 = self.tform.inverse((self.trail_deque_dict[id][0 ][0] , self.trail_deque_dict[id][0 ][1]  ))
                x1  = (int(p1[0][0]) , int(p1[0][1]))
                x2  = (int(p2[0][0]) , int(p2[0][1]))
                d = cv2.line(zeros,(x1),(x2) , 200 , 20)
                #print(x1,x2,'\n')
                res = np.add( d ,res )

        res = cv2.GaussianBlur(res,(5,5), 2)

        heatmap_img = cv2.applyColorMap(res, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.3, img, 0.7,0)
        return res , super_imposed_img

                



    def _draw_Trails(self,img, img1 , bbox, object_id, identities=None ):
        # remove tracked point from buffer if object is lost
        for key in list(self.trail_deque_dict):
            if key not in identities:
                self.trail_deque_dict.pop(key)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]

            # code to find center of bottom edge
            center = (int((x2 + x1) / 2), int(y2))
            # get ID of object
            id = int(identities[i]) if identities is not None else 0
            color = self._compute_color_for_labels(object_id[i])
            

            # create new buffer for new object
            if id not in self.trail_deque_dict:
                self.trail_deque_dict[id] = deque(maxlen=self.trail_max_length)

            
            # add center to buffer
            self.trail_deque_dict[id].appendleft(center)
            
            # draw trail
            for i in range(1, len(self.trail_deque_dict[id])):
                # check if on buffer value is none
                if self.trail_deque_dict[id][i - 1] is None or self.trail_deque_dict[id][i] is None:
                    continue

                # generate dynamic thickness of trails
                thickness = int(np.sqrt(self.trail_max_length / float(i + i)) * 1.5)

                # draw trails
                cv2.line(img, self.trail_deque_dict[id][i - 1], self.trail_deque_dict[id][i], color, thickness)

                #draw trails on map
                pts2_ = self.tform.inverse((self.trail_deque_dict[id][i - 1][0], self.trail_deque_dict[id][i - 1][1] ))
                pts3_ = self.tform.inverse((self.trail_deque_dict[id][i ][0] , self.trail_deque_dict[id][i ][1]  ))
                x1_ = (int(pts2_[0][0]), int(pts2_[0][1]))
                x2_ = (int(pts3_[0][0]), int(pts3_[0][1]))
                cv2.line(img1, x1_, x2_, color, thickness)

                

        return img
    
    def _save_restults(self,frame, bbox, identities):
        txt_path = 'res.txt'
        print(bbox)
        for i, box in enumerate(bbox):
            id = identities[i]
            save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
            with open(txt_path , 'a') as f:
                    line = save_format.format(frame=frame, id=id, x1=int(box[0]), y1=int(box[1]), w=int(box[2]- box[0]), h=int(box[3]-box[1]), x = -1, y = -1, z = -1)
                    f.write(line)



    def Track_and_estimate_Speed(self,kpi1_text, kpi2_text, stframe , stframe1 , stframe3 , stframe4):

        self.trail_deque_dict = {}
        map = np.asarray(self.map)
        map1 = map.copy()
    
        dataset = LoadImages(self.source, img_size=self.imagesize, auto_size=64)
        # Run inference
        t0 = time.time()
        
        prevTime = 0

                
        cookie = Cookie()
        la1 , lo1 ,la2 , lo2  = cookie.get("coords")
        long = (lo2 + lo1)/2
        lat = (la2 + la1)/2
        res = np.zeros((512,640),np.uint8)
        super_imposed_img = None
        
        for path, img, im0, vid_cap ,frame in dataset:
            #detect
            xywhs, confss, oids = self.detector.detect(img,im0)
            
            #update Trackor
            if xywhs is not None:
                outputs = self.deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    #draw trails
                    self._draw_Trails(im0,map, bbox_xyxy, object_id, identities)
                    #estimate_speed
                    self._estimate_Speed(im0, bbox_xyxy, object_id, identities)
                    #get on map
                    self._drawonmap(stframe3, bbox_xyxy, object_id, identities)
                    #get heatmap
                    
                    res , super_imposed_img = self._generateHeat(map1 , res, bbox_xyxy, identities)

                    #save_results
                    #self._save_restults( frame , bbox_xyxy, identities)

            #calculate FPS
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            #return images to streamlit frames
            stframe.image(im0 , channels= 'BGR' , use_column_width = 3 )
            stframe3.image( map  , channels= 'BGR' , use_column_width = 3 )
            if self.dict['lat'] is not None:
                df = pd.DataFrame(data=self.dict,dtype=np.float64)
                stframe1.pydeck_chart(pdk.Deck(
                                        map_style=None,
                                        initial_view_state=pdk.ViewState(
                                            latitude=lat,
                                            longitude=long,
                                            zoom=16,
                                            pitch=30,
                                        ),
                                        layers=[
                                            pdk.Layer(
                                                'ScatterplotLayer',
                                                data=df,
                                                get_position='[lon, lat]',
                                                get_color='[200, 30, 0, 160]',
                                                get_radius=5,
                                            ),
                                        ],
                                    ))
            if super_imposed_img is not None:
                stframe4.image(super_imposed_img , channels= 'BGR' , use_column_width = 3)
            kpi1_text.write( f"<h1 style =' color: red ; ' > {'{:.1f}'.format(fps)} </h1> ",unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style ='color: red ; ' > {'{}'.format(len(self.trail_deque_dict.keys()))} </h1> ",unsafe_allow_html = True)
        print('Done. (%.3fs)\n' % (time.time() - t0))