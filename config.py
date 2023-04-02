import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from keras.models import load_model
from PIL import Image
import numpy as np
from playsound import playsound
import random

model_name = 'fer_resnet_model4'
inpDir = 'Dataset/FER' # location where input data is stored
outDir = 'OUTPUT_DIRECTORY/' # location to store outputs where save the model and weights
casDir = r'C:\Users\AKASH\Desktop\common_project\FER\haarcascades'

#AgeDir =  'AGE'# location of the images of data related file
#GenderDir = 'GENDER' # location to data related file
#FerDir = 'FER'

trainDir = 'train'
testDir = 'test'

modelDir = 'models' # location to save model files
HappySongDir = 'songs/0'
SadSongDir = 'songs/1'

inferenceDir = r'inference/a' # location related to this dataset
weights=r"models/fer_resnet_model4.h5"



RANDOM_STATE = 24 # for initialization ----- REMEMBER: to remove at the time of promotion to production

tf.random.set_seed(RANDOM_STATE)



EPOCHS = 2  # number of cycles to run
ALPHA = 0.01
BATCH_SIZE = 16
TEST_SIZE = 0.001
TEST_SIZE_VAL = 0.9999
IMG_HEIGHT = 224
IMG_WIDTH = 224
FLIP_MODE = "horizontal_and_vertical"
# for rotation transformation 
ROTATION_FACTOR = (-0.1, 0.1) 
FILL_MODE = 'nearest'
ES_PATIENCE = 20 # if performance does not improve stop
LR_PATIENCE = 10 # if performace is not improving reduce alpha
LR_FACTOR = 0.5 # rate of reduction of alpha# Train the model
num_epochs = 10
# Set parameters for decoration of plots
params = {'legend.fontsize' : 'large',
          'figure.figsize'  : (15,6),
          'axes.labelsize'  : 'x-large',
          'axes.titlesize'  :'x-large',
          'xtick.labelsize' :'large',
          'ytick.labelsize' :'large',
         }
CMAP = plt.cm.brg
plt.rcParams.update(params) # update rcParams
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

save_model = 'age_resnet_model1.h5'


#data_dir = os.path.join(inpDir, FerDir,trainDir)
data_dir = os.path.join(inpDir,trainDir)
#data_dir2 = os.path.join(inpDir, FerDir , testDir)
data_dir2 = os.path.join(inpDir, testDir)
