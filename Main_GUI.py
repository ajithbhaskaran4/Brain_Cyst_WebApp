import streamlit as st
from BackEnd.Image2PointCloud import Image2PointCloud, CNN_Prediction
from stpyvista import stpyvista
import pyvista as pv
import  streamlit_vertical_slider  as svs
from PIL import Image
import numpy as np
import tempfile
import os
import io


pv.set_jupyter_backend('static')

st.set_page_config(layout="wide")


# ipythreejs does not support scalar bars :(
pv.global_theme.show_scalar_bar = False

if 'flag' not in st.session_state:
    st.session_state.flag = False
    
if 'CNN' not in st.session_state:
    st.session_state.CNN = CNN_Prediction()

if 'sliderPos' not in st.session_state:
    st.session_state.sliderPos = 0
    
if 'backend' not in st.session_state:
    st.session_state.backend = Image2PointCloud()
st.title("MRI CYST ANALYSIS")
st.text("An Application to view and quantify the presence of cyst from MRI")


def change_MRI():
    #st.subheader("MRI Image")
    currentImage = st.session_state.ImageStack[st.session_state.MRI_Slider, :,:,:]
    #print("current Image size: ", currentImage.shape)
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
    if 'prediction' not in st.session_state:
        st.session_state.prediction = st.session_state.CNN.predictCNN(st.session_state.ImageStack)
       
    print("received all data")
    st.session_state.flag = True



if st.session_state.flag == True:
    st.session_state.sliderPos = st.slider("Select MRI Image Slice", min_value=0, max_value=st.session_state.NumImages, step=1, key = "MRI_Slider")#,on_change=change_MRI)    
    currentImage = Image.fromarray(np.uint8(st.session_state.backend.getMRIImage(st.session_state.sliderPos)), mode = "RGB")
    st.image(currentImage)
    st.write(st.session_state.prediction[st.session_state.sliderPos,:,:,0].shape)
    currentPred = Image.fromarray(st.session_state.prediction[st.session_state.sliderPos,:,:,0], mode = 'L')
    st.image(currentPred)
            