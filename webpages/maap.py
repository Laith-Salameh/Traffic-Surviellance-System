import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

d = {'lat': [33.57184532469701 ], 'lon': [36.235991213086244] }
df = pd.DataFrame(data=d)

print(df['lat'][0])

st.pydeck_chart(pdk.Deck(
     map_style=None,
     initial_view_state=pdk.ViewState(
         latitude=36.23,
         longitude=33.571,
         zoom=11,
         pitch=50,
     ),
     layers=[
         pdk.Layer(
             'ScatterplotLayer',
             data=df,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 160]',
             get_radius=10,
         ),
     ],
 ))