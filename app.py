import streamlit as st

from streamlit_option_menu import option_menu


from webpages import map,  index, transform , track

st.set_page_config(layout="wide")



if 'pages' not in st.session_state.keys():
    st.session_state['pages'] = { 'Home' : ( index.app , 'house' ),
    'Map' : (map.app , 'geo'),
        'Transform' : (transform.app , 'arrow-left-right'),
        'Track': (track.app , 'car-front' )
    }
if 'current page' not in st.session_state.keys():
    st.session_state['current page'] = 0
icons = list( st.session_state['pages'].values())

col1 ,col2 = st.columns([10,1])
with col1:
    selected = option_menu(
        None, list(st.session_state['pages'].keys()), 
        icons= [ x[1] for x in icons] , 
        menu_icon="cast", default_index=st.session_state['current page'], orientation="horizontal")


st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
menu { width : 100wh; }
</style> """, unsafe_allow_html=True)

st.session_state['pages'][selected][0]()
