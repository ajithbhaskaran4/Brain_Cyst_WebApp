import numpy as np
import os
from PIL import Image
import pyvista as pv
import natsort 
import matplotlib.pyplot as plt
pv.set_jupyter_backend('static')



cmap = plt.get_cmap('viridis')


class Image2PointCloud:
    def __init__(self):
        self.directory = [] #r'P:\MRI_Cyst\archive\kaggle_3m\TCGA_HT_A61A_20000127'  #TCGA_HT_A61A_20000127 TCGA_HT_8107_19980708
        self.images = np.empty((0, 256, 256, 3))
        self.pointcloud = []
        self.color = []
        
    def setPaths(self, path):
        self.directory = path
        
    def read_mri_images(self):
        self.images = np.empty((0, 256, 256, 3))
        directoryList = os.listdir(self.directory)
        #print("Unsorted : ",directoryList)
        directoryList = natsort.natsorted(directoryList,reverse=False)
        #print("Sorted: ", directoryList)
        for filename in os.listdir(self.directory):
            if not filename.endswith("_mask.tif"): 
                #print("Image: ", filename)
                img_path = os.path.join(self.directory, filename)
                img = np.asarray(Image.open(img_path))  # Read the image in grayscale
                img = np.expand_dims(img, axis=0)
                self.images = np.append(self.images, img, axis = 0)
                
    def getnumberofImages(self):
        return self.images.shape[0]
                
    def convert2PointCloud(self):
        points = []
        colors = []
        transparency = []
        scale_factor = 0.1
        
        points = np.argwhere(self.images[:,:,:,1])
        colors = cmap(self.images[points[:, 0], points[:, 1], points[:, 2], :]/255)
        transparency = self.images[points[:, 0], points[:, 1], points[:, 2], 1]/255
        
        points = np.array(points)
        print("point shape: ", points.shape)
        transparency = np.array(transparency).astype(float)
        print("transparency shape: ", transparency.shape)
        colors = np.array(colors)
        print("colors shape: ", colors.shape)
        
        print("Point Cloud Completed")
        self.point_cloud = pv.PolyData(points)
        print("Point Cloud color")
        self.point_cloud['point_color'] = colors
        print("Point Cloud transparecy")
        self.point_cloud['transparency'] = transparency
        #mesh = self.point_cloud.delaunay_3d()
        print("meshing completed")
        return self.point_cloud
                   
        
    def get_StackMRI(self):
        print("Image stack Size: ", self.images.shape)
        return self.images
    