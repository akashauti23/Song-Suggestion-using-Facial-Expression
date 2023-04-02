import warnings
warnings.filterwarnings("ignore")

from model import *
from config import *

#train the data according to the age and gender 
# dataloader file
train_ds_fer,test_ds_fer,class_name_fer=dataloader(data_dir,data_dir2)  

#build model
resnet_model=build_model(input_shape, class_name_fer ,train_ds_fer,test_ds_fer) 

# compile model
compile_model(resnet_model,train_ds_fer,test_ds_fer) 

#os.getcwd()      
#os.chdir('C:/Users/Atharva Kadam/OneDrive/Desktop/common_project/models')

resnet_model.save(modelDir+"/"+model_name+"_test.h5") # saving the model
