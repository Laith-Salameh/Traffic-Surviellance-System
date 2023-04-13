import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from Transformer import *
from Cookie import *

def app():

    drawing_mode = "point"

    stroke_width = 1
    point_display_radius = 3

    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = "#eee" 

    cookie = Cookie()
    sat_img = cookie.get('sat_image')
    cam_img = cookie.get('background_image')
    realtime_update = True


    if sat_img is not None:
        sat_img = Image.open(sat_img)
    if cam_img is not None:
        cam_img = Image.open(cam_img)

    transformer = Transformer(sat_img , cam_img)


    col1 , col2 = st.columns(2)

    with col1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color= stroke_color, 
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image= sat_img,
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            height =512,
            width = 640,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )
        if sat_img is None:
            st.error('upload satellite image')
    with col2:
        canvas_result1 = st_canvas(
            fill_color=stroke_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image= cam_img,
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            height =512,
            width = 640,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas1",
        )
    if cam_img is  None:
        st.error("upload video")

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")

        if 'left' in objects.keys():
            for x , y in zip(objects['left'],objects['top']):
                transformer.append_Sat_points((x,y))

    if canvas_result1.json_data is not None:
        objects1 = pd.json_normalize(canvas_result1.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects1.select_dtypes(include=['object']).columns:
            objects1[col] = objects1[col].astype("str")

        if 'left' in objects1.keys():
            for x , y in zip(objects1['left'],objects1['top']):
                transformer.append_Cam_points((x,y))


        if cam_img is not None and sat_img is not None:
            tform , tf_img = transformer.Homogenous_Transform()
            if tform is not None:
                st.image(tf_img , channels= 'RGB' , use_column_width = 3 ,clamp=True)
                conf = st.button("Confirm")
                if conf:
                    
                    cookie.set("sat_points",transformer.get_sat_points())
                    cookie.set("cam_points",transformer.get_cam_points())
                    st.success('saved')



