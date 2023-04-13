from distutils.command.config import config
import streamlit as st
from Trackor import *
import torch
import io
from Cookie import *
from Transformer import get_Homogenous_Transform_from_Cookie

def app():
    cookie = Cookie()
    
    names = ['other' , 'car' , 'van' , 'bus']
    assigned_class_id = []
    sat_or_road_img = None
    
    kpi1 , kpi2 , kpi3 = st.columns(3)
    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')
    with kpi2:
        st.markdown('**Objects being tracked**')
        kpi2_text = st.markdown('0')
    with kpi3:
        with st.expander("Track Custom Classes"):
            assigned_class = st.multiselect("Select the Custom Classes", list(names))
            for each in assigned_class:
                assigned_class_id.append(names.index(each))


    confidence = st.sidebar.slider('confidence',min_value= 0.0, max_value=1.0,value=0.7) 
    IOUmin = st.sidebar.slider('IOUmin',min_value= 0.0, max_value=1.0,value=0.7)

    sat_or_road_img =  None

    
    col11 , col22 = st.columns(2)
    with col11:
        stframe = st.empty()
    with col22:
        stframe1 = st.empty()

    col1_1 , col2_1 = st.columns(2)
    with col1_1:
        stframe3 = st.empty()
    with col2_1:
        stframe4 = st.empty()

    start = st.button("start tracking")

    video = cookie.get("video")
    sat_or_road_img = cookie.get("sat_image") if sat_or_road_img is None else None
    img = cv2.imread(sat_or_road_img)
    tfrom = get_Homogenous_Transform_from_Cookie()
    #intialiaze Trackor
    if video is not None and img is not None and tfrom is not None and start:
        trackor = Trackor( names , video ,confidence,IOUmin,assigned_class_id ,tfrom, img)
        with torch.no_grad():
            trackor.Track_and_estimate_Speed(  kpi1_text,kpi2_text , stframe , stframe1 ,stframe3 , stframe4)
    

  


    
