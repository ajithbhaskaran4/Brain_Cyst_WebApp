import numpy as np
import os
from PIL import Image
import pyvista as pv
import natsort 
import matplotlib.pyplot as plt

import keras
import numpy as np
import os
from keras import backend as K

pv.set_jupyter_backend('static')



cmap = plt.get_cmap('viridis')

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


class Image2PointCloud:
    def __init__(self):
        self.directory = [] #r'P:\MRI_Cyst\archive\kaggle_3m\TCGA_HT_A61A_20000127'  #TCGA_HT_A61A_20000127 TCGA_HT_8107_19980708
        self.images = np.empty((0, 256, 256, 3))
        self.pointcloud = []
        self.color = []
        
    def setPaths(self, path):
        self.directory = path
        
    def getMRIImage(self, Image_Number):
        return self.images[Image_Number,:,:,:]
        
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
        
class CNN_Prediction():
    def __init__(self):
        self.ModelPath = r'BackEnd/Unet_Best_Model.hdf5'
        self.Model = keras.models.load_model(self.ModelPath,custom_objects={"dice_coef": dice_coef,"dice_loss": dice_loss })
        self.mean = 21.77118
        self.std = 32.471928
        
    def predictCNN(self, Input):
        Input = Input.astype('float32')
        Input = (Input - self.mean) / self.std
        Pred = model.predict(Input)
        Pred = (Pred > 0.5).astype('uint8')
        return Pred
    