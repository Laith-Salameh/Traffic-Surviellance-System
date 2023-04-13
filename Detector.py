from utils.torch_utils import select_device

from models.models import *
from utils.datasets import *
from utils.general import *
from utils.torch_utils import time_synchronized

class Detector:
    def __init__(self,names , confidence , IOU_threshold, assigned_class_id , weights, imgsize, cfg):
        self.names = names
        self.confidence = confidence
        self.IOU_threshold = IOU_threshold
        self.assigned_class_id = assigned_class_id
        self.weights = weights
        self.imgsize = imgsize
        self.cfg = cfg
        self.device = select_device('0')
        # Load model
        self.model = Darknet(self.cfg, (self.imgsize , self.imgsize)).cuda()
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        self.model.to(self.device).eval()
        self.model.half()  # to FP16
        img = torch.zeros((1, 3, self.imgsize , self.imgsize ), device=self.device)  # init img
        _ = self.model(img.half()) # run once
    
    
    def xyxy_to_xywh(self,*xyxy):
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h
    
    def detect(self, img ,im0 ):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() 
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.confidence, self.IOU_threshold, classes=self.assigned_class_id)
        t2 = time_synchronized() 
        s = ''
        det = pred[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            xywh_bboxs = []
            confs = []
            oids = []
            for *xyxy, conf, cls in det:
                # to deep sort format
                x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))

                
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            return xywhs , confss , oids
        else:
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            return None,None,None
        