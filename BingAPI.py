import requests
from PIL import Image
import numpy as np
import io

class BingAPI:
    def __init__(self):
        self.ppm_forzoom = {
            1:	1/78271.52,
            2:	1/39135.76,
            3:	1/19567.88,
            4:	1/9783.94,
            5:	1/4891.97,
            6:	1/2445.98,
            7:	1/1222.99,
            8:	1/611.50,
            9:	1/305.75,
            10:	1/152.87,
            11:	 1/76.44,
            12:	1/38.22,
            13:	1/19.11,
            14:	1/9.55,
            15:	1/4.78,
            16:	1/2.39,
            17:	1/1.19,
            18:	1/0.60,
            19:	1/0.30
        }
    def getppm(self , zoom):
        return self.ppm_forzoom[zoom]
        
    def _getImage(self, imagerySet , x1 , y1 ,x2,y2,zoomlevel , mapLayer = None):
        
        pushpin = '{},{};0;2'.format(x2,y2)
        URL = ("https://dev.virtualearth.net/REST/v1/Imagery/Map/{}/?pp={}&dcl=1").format(imagerySet,pushpin)
        mapArea = '{},{},{},{}'.format(x1,y1,x2,y2)
        zoomLevel = zoomlevel
        pp = '{},{};0;1'.format(x1,y1)
        mapMetadata = 0
        PARAMS = {  'mapArea' : mapArea,
                    'mapMetadata' : mapMetadata,
                    'zoomLevel': zoomLevel,
                    'pp': pp,
                    'key' : 'AoWrac7Uz5AnBDNO-7oLO13QvBKsE_8pYSQaeIohNKijyExifefQUEfwvQ1Czy_z' }
        if mapLayer:
            PARAMS['mapLayer'] = mapLayer
        data = requests.get(url = URL, params = PARAMS)
        if data.status_code == 200:
            img = Image.open(io.BytesIO(data.content))
            PARAMS['mapMetadata'] = 1
            result = requests.get(url = URL, params = PARAMS)
            if result.status_code ==200:
                metadata = result.json()
                xx = np.empty(0,dtype=np.uint8)
                yy = np.empty(0,dtype=np.uint8)
                for pp in metadata['resourceSets'][0]['resources'][0]['pushpins']:
                    x1 = int(pp['anchor']['x'])
                    y1 = int(pp['anchor']['y'])
                    xx = np.append(xx,x1)
                    yy = np.append(yy,y1)

                cropped_im = img.crop((xx.min(), yy.min() , xx.max() , yy.max()))
                cropped_im = cropped_im.convert("RGB")
                return cropped_im
           

        print(data.status_code)
        return None

    def getSatImages(self,x1,y1,x2,y2,zoomlevel):
        imagerySet = "Aerial"
        return self._getImage(imagerySet, x1 ,y1,x2,y2,zoomlevel)
        
    def getRoadImages(self,x1,y1,x2,y2,zoomlevel):
        imagerySet = "Road"
        mapLayer = 'Basemap,Buildings'
        return self._getImage(imagerySet, x1 ,y1,x2,y2,zoomlevel,mapLayer )