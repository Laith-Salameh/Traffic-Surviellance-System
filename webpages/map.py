import geemap.foliumap as geemap
import streamlit as st
from PIL import Image
import numpy as np
from BingAPI import *
from Cookie import *


import ee
service_account = 'laith-95@metasearch-1646855910305.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'key.json')
ee.Initialize(credentials)

def app():
        cookie = Cookie()
        col1_1 ,  col2_1 , col3_1 , col4_1 , col5_1 = st.columns([2,2,2,1,1])
        with col1_1:
            lon = st.slider('Longitude',min_value= -180.0, max_value=180.0,value= 0.0 )
        with col2_1:
            lat = st.slider('Latitude',min_value= -90.0, max_value=90.0,value= 0.0 )
        with col3_1:
            zoom = st.number_input(max_value=19 , min_value=0 , label='Zoom' , value= 4)

            
    
        bingapi = BingAPI()
        col1_2 ,  col2_2 = st.columns([3,1])
        with col1_2:
                Map = geemap.Map(plugin_Draw=True, Draw_export=False)
                Map.add_basemap('HYBRID')
                Map.setCenter(lon, lat, zoom)
                st_comp = Map.to_streamlit(bidirectional=True)

        with col2_2:
            zoom = st.number_input("Zoom level",min_value=10 , max_value=19 , value=15)
            getImage = st.button('Get Drawn Region')
            sat_img_name = 'images/bing_sat_img.png'
            road_img_name = 'images/bing_road_img.png'
            draw_feature = Map.st_last_draw(st_component=st_comp)
            scale_w = 1
            scale_h= 1
            if getImage and draw_feature is not None:
                if draw_feature['geometry']['type'] == 'Polygon':
                    a = np.float32(draw_feature['geometry']['coordinates'][0])
                    y1 , x1 = a.reshape(-1 , 1 ,2 ).min(axis=0).ravel()
                    y2 , x2 = a.reshape(-1 , 1 ,2 ).max(axis=0).ravel()
                    #try:
                    sat_img = bingapi.getSatImages(x1,y1,x2,y2,zoom)
                    cookie.set("coords" , list([ float(x1) , float(y1) , float(x2) , float(y2)]) )
                    #except:SS
                        #st.error("Select smaller region")
                        #return 
                    st.image(sat_img)
                    #road_img = bingapi.getRoadImages(x1,y1,x2,y2,zoom)
                    size = ( 640,512  )
                    width , height = sat_img.size
                    scale_w = width/size[0]
                    scale_h = height/size[1] 

                    sat_image = sat_img.resize(size)
                    #road_img = road_img.resize(size)
                    sat_image.save(sat_img_name)
                    cookie.set("scale_w" , scale_w )
                    cookie.set("scale_h" ,  scale_h )
                    #road_img.save(road_img_name)


                else:
                    st.error("Use Polygon")
            save = st.button("Use Satelite Image")
            if save:
 
                cookie.set('sat_image' , sat_img_name)
                #cookie.set('road_image' , road_img_name)
                cookie.set('ppm' , bingapi.getppm(zoom))
                st.success("done")

                
