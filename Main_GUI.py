import streamlit as st
from BackEnd.Image2PointCloud import Image2PointCloud
from stpyvista import stpyvista
import pyvista as pv
import  streamlit_vertical_slider  as svs
from PIL import Image
import numpy as np
import tempfile
import os
import io
from ipywidgets import embed
import streamlit.components.v1 as components
from pyvista.jupyter.pv_pythreejs import convert_plotter

pv.start_xvfb()

pv.set_plot_theme('document')

pv.set_jupyter_backend('static')

st.set_page_config(layout="wide")

def pyvista_streamlit(plotter):
    widget = convert_plotter(plotter)
    state = embed.dependency_state(widget)
    fp = io.StringIO()
    embed.embed_minimal_html(fp, None, title="", state=state)
    fp.seek(0)
    snippet = fp.read()
    components.html(snippet, width=900, height=500)

# ipythreejs does not support scalar bars :(
pv.global_theme.show_scalar_bar = False

if 'flag' not in st.session_state:
    st.session_state.flag = False
    
if 'backend' not in st.session_state:
    st.session_state.backend = Image2PointCloud()
st.title("MRI CYST ANALYSIS")
st.text("An Application to view and quantify the presence of cyst from MRI")

def change_MRI():
    with MRICol:
        st.subheader("MRI Image")
        currentImage = st.session_state.ImageStack[st.session_state.MRI_Slider, :,:,:]
        print("current Image size: ", currentImage.shape)
        currentImage = Image.fromarray(np.uint8(currentImage), mode = "RGB")
        st.image(currentImage)
               
uploaded_files  = st.file_uploader("Choose the MRI Images", accept_multiple_files=True, key = 'Folder_Path')

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    st.write("Uploaded Files:")
    for file in uploaded_files:
        file_name = file.name
        file_path = os.path.join(temp_dir, file_name)
        # Save the uploaded file with its original name to the temporary directory
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.session_state.backend.setPaths(temp_dir)


if st.button('Submit'):
    st.session_state.backend.read_mri_images()
    #st.session_state.pointCloud = backend.convert2PointCloud()
    #st.session_state.NumImages = backend.getnumberofImages()
    #st.session_state.ImageStack = backend.get_StackMRI()
    if 'pointCloud' not in st.session_state:
        st.session_state.pointCloud = st.session_state.backend.convert2PointCloud()
    if 'NumImages' not in st.session_state:
        st.session_state.NumImages = st.session_state.backend.getnumberofImages()
    if 'ImageStack' not in st.session_state:
        st.session_state.ImageStack = st.session_state.backend.get_StackMRI()
    print("received all data")
    st.session_state.flag = True



if st.session_state.flag == True:

    pointCol, intermediate, MRICol = st.columns([3, 1,3])
    with pointCol:
        st.subheader("3D MRI View")
        st.slider("Select MRI Image Slice", min_value=0, max_value=st.session_state.NumImages, step=1, key = "MRI_Slider",on_change=change_MRI)
        plotter = pv.Plotter(window_size=[600,600])
        plotter.add_points(st.session_state.pointCloud, opacity = st.session_state.pointCloud['transparency'], cmap='bone') #, opacity = pointCloud['transparency']
        plotter.add_scalar_bar()
        plotter.view_isometric()
        plotter.background_color = 'white'
        #stpyvista(plotter, key="MRI")
        pyvista_streamlit(plotter)
        
            