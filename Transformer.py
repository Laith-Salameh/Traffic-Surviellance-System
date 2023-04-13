from skimage import transform 
import numpy as np
from PIL import Image
from Cookie import *

class Transformer:
    def __init__(self,sat_image,cam_image):
        self.sat_image = sat_image
        self.cam_image = cam_image
        self.sat_points = []
        self.cam_points = []

    def append_Sat_points(self,point):
        self.sat_points.append(point)

    def append_Cam_points(self,point):
        self.cam_points.append(point)
    
    def get_sat_points(self):
        return self.sat_points
    def get_cam_points(self):
        return self.cam_points
    
    def Homogenous_Transform(self):
        if( len(self.sat_points) >=4 and len(self.cam_points)>=4  and len(self.sat_points) == len(self.cam_points)):
            p1 = np.float32( self.sat_points)
            p2 = np.float32(self.cam_points)
            tform = transform.estimate_transform('projective', p1, p2)
            np_cam_img = np.asarray(self.cam_image)
            tf_img = transform.warp(np_cam_img, tform ,output_shape=(512,640))
            tf_img_pil = Image.fromarray((tf_img * 255).astype(np.uint8))
            tf_img_pil = tf_img_pil.convert("RGBA")
            sat_img_pil = self.sat_image.convert("RGBA")
            print(tf_img_pil.mode , self.sat_image.mode  )
            blended = Image.blend(tf_img_pil, sat_img_pil , alpha=0.5)
            return tform , blended

        return None , None

def get_Homogenous_Transform_from_Cookie():
    cookie = Cookie()
    sat_points = cookie.get('sat_points')
    cam_points = cookie.get('cam_points')
    if sat_points is not None and cam_points is not None:
        p1 = np.float32(sat_points)
        p2 = np.float32(cam_points)
        tform = transform.estimate_transform('projective', p1, p2)
        return tform
    else:
        return None


