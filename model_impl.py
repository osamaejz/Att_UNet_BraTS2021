import os
import numpy as np
# from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras


## Custom Data Generator
def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

############
            
training_image_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Training_set/image/'
training_mask_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Training_set/mask/'
training_images = os.listdir(training_image_directory)
training_masks = os.listdir(training_mask_directory)

testing_image_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Testing_set/image/'
testing_mask_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Testing_set/mask/'
testing_images = os.listdir(testing_image_directory)
testing_masks = os.listdir(testing_mask_directory)

val_image_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Validation_set/image/'
val_mask_directory = 'F:/Osama DELL NCL PC/Att_UNet Data/Validation_set/mask/'
val_images = os.listdir(val_image_directory)
val_masks = os.listdir(val_mask_directory)


batch_size = 5

train_img_datagen = imageLoader(training_image_directory, training_images, 
                                training_mask_directory, training_masks, batch_size)

val_img_datagen = imageLoader(val_image_directory, val_images, 
                                val_mask_directory, val_masks, batch_size)



IMG_HEIGHT = 128
IMG_WIDTH  = 128
IMG_DEPTH  = 128
IMG_CHANNELS = 3
#num_labels = 1  #Binary
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS)

import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25,0.25,0.25,0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
#######################################################################
#Fit the model 

steps_per_epoch = len(training_images)//batch_size
val_steps_per_epoch = len(val_images)//batch_size




from Attention_U_Net import Attention_UNet, dice_coef, dice_coef_loss, jacard_coef

att_unet_model = Attention_UNet(input_shape)

att_unet_model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(att_unet_model.summary())



history=att_unet_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

att_unet_model.save('brats_3d.hdf5')







