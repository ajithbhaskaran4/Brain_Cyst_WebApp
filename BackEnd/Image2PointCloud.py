import numpy as np
import os
from PIL import Image, ImageFilter
import pyvista as pv
import natsort 
import matplotlib.pyplot as plt

import keras
import numpy as np
import os
from keras import backend as K

pv.set_jupyter_backend('static')



cmap = plt.get_cmap('brg')

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


class CNN_Prediction():
    def __init__(self):
        self.ModelPath = r'BackEnd/Unet_Best_Model.hdf5'
        self.Model = keras.models.load_model(self.ModelPath,custom_objects={"dice_coef": dice_coef,"dice_loss": dice_loss })
        self.mean = 21.77118
        self.std = 32.471928
        self.pred = []
        
    def predictCNN(self, Input):
        Input = Input.astype('float32')
        Input = (Input - self.mean) / self.std
        Pred = self.Model.predict(Input)
        Pred = (Pred > 0.5).astype('uint8')
        self.pred = Pred*255
        return self.pred
    
    def getCystPointCloud(self):
        predPoint = np.argwhere(self.pred[:,:,:,0]==255)
        redColor = np.array([0,1,0]) # DIMENSION = (3,)
        redColor = np.expand_dims(redColor, axis=0) #dIMENSION = (1,3)
        predcolors = cmap(np.repeat(redColor, predPoint.shape[0], axis = 0)) #DIMENSION = (NUMBER OF POINTS, 3)
        predtransparency = np.zeros(predPoint.shape[0])
        
        self.cycst_cloud = pv.PolyData(predPoint)
        predcolors = np.array(predcolors)
        self.cycst_cloud['point_color'] = predcolors
        print("Point Cloud transparecy")
        predtransparency = np.array(predtransparency).astype(float)
        self.cycst_cloud['transparency'] = predtransparency
        return self.cycst_cloud
        

class Image2PointCloud:
    def __init__(self):
        self.directory = [] #r'P:\MRI_Cyst\archive\kaggle_3m\TCGA_HT_A61A_20000127'  #TCGA_HT_A61A_20000127 TCGA_HT_8107_19980708
        self.images = np.empty((0, 256, 256, 3))
        self.edges = np.empty((0, 256, 256, 1))
        self.pointcloud = []
        self.color = []
        
    def setPaths(self, path):
        self.directory = path
        
    def getMRIImage(self, Image_Number):
        return self.images[Image_Number,:,:,:]
        
    def read_mri_images(self):
        self.images = np.empty((0, 256, 256, 3))
        self.edges = np.empty((0, 256, 256, 1))
        directoryList = os.listdir(self.directory)
        #print("Unsorted : ",directoryList)
        directoryList = natsort.natsorted(directoryList)#,reverse=False)
        #print("Sorted: ", directoryList)
        for filename in directoryList:
            if not filename.endswith("_mask.tif"): 
                #print("Image: ", filename)
                img_path = os.path.join(self.directory, filename)
                img = Image.open(img_path)
                img = img.resize((256,256))
                edge = img.convert("L")
                edge = edge.filter(ImageFilter.FIND_EDGES)
                edge = np.expand_dims(np.asarray(edge), axis=0)
                edge = np.expand_dims(edge, axis=-1)
                self.edges = np.append(self.edges, edge, axis = 0)
                img = np.asarray(img)  # Read the image in grayscale
                img = np.expand_dims(img, axis=0)
                self.images = np.append(self.images, img, axis = 0)
                
    def getnumberofImages(self):
        return self.images.shape[0]
        
                
    def convert2PointCloud(self):
        points = []
        colors = []
        transparency = []
        scale_factor = 0.1
        
        points = np.argwhere(self.edges[:,:,:,0]>=30)
        colors = cmap(self.images[points[:, 0], points[:, 1], points[:, 2], :]/255)
        transparency = np.ones(points.shape[0])*0.01
       
        
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
        
    