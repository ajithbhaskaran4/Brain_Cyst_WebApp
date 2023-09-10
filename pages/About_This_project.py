import streamlit as st
from BackEnd.Image2PointCloud import Image2PointCloud, CNN_Prediction
from PIL import Image
import os


st.title("MRI CYST ANALYSIS")
st.text("An Application to view and quantify the presence of cyst from MRI")
st.write()
st.write()

st.write("Brain MRI (Magnetic Resonance Imaging) is a non-invasive medical imaging technique that plays a pivotal role in the early identification and diagnosis of various neurological conditions. When combined with advanced image processing algorithms, MRI scans become powerful tools for the detection of cysts within the brain. These algorithms can analyze the intricate details of the brain's structure, helping to differentiate cysts from normal tissue and providing valuable information about their size, location, and characteristics. This technology is of paramount importance in clinical practice as it enables early intervention and treatment planning for patients with brain cysts. Timely identification and accurate characterization of these abnormalities can significantly improve patient outcomes, reduce the risk of complications, and enhance the quality of life for those affected by neurological conditions.")