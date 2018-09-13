# -*- coding: utf-8 -*-

# CNN Dog Cat Dataset
# Building CNN 

#Importing the keras libraries and packages
from keras.models import Sequential    # Sequential Model
from keras.layers import Conv2D # cnn layer
from keras.layers import MaxPooling2D  # pooled feature map
from keras.layers import Flatten       # to flatten the vector to 1-D that can 
                                       # act as input to a FC
from keras.layers import Dense         # Dense Layer 
from keras.preprocessing.image import ImageDataGenerator

#Reference Keras Documentation Image Processing ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, #all pixel values between 0 & 1 similar to feature scaling
        shear_range=0.2,
        zoom_range=0.2, #random zoom
        horizontal_flip=True) #flipped horizontally

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Initializing the CNN
def build_model():
    model = Sequential()
    
    #step 1 convolution
    #using tensorflow backend on CPU hence using 64 x64 pixels 3  (RGB)
    #creating 32 feature maps of 3 x 3 
    #non linearlity relu
    model.add(Conv2D(32, (3, 3), input_shape =(64,64,3) , activation='relu')) 
    
    #step 2 max pooling , reduce feature maps to Pooled feature map
    model.add(MaxPooling2D(pool_size= (2,2)))
    
    #Adding second convolution layer and max pooling, since we have Pooled
    # feature map , we dont need to specify input_shape
    model.add(Conv2D(32, (3, 3), activation='relu')) 
    model.add(MaxPooling2D(pool_size= (2,2)))    
    
    #step 3 Flattening, taking the pooling layer and making 1-D
    model.add(Flatten()) # can be used as input to a FC
    
    
    #step 4 FC Layer
    #common practice to pick a power of 2
    model.add(Dense(activation='relu', units= 128))
    model.add(Dense(activation='sigmoid', units= 1)) # output if more than 2 categories then softmax

    #Compiling the cnn
    # sgd, loss fxn and performance metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model()

model.fit_generator(
        training_set,
        steps_per_epoch=8000, 
        epochs=2, #running on CPU , hence only 2 epochs try with 25 for much higher accuracy
        validation_data=test_set,
        validation_steps=2000)    


##Predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

result = model.predict(test_image)
print(result)

training_set.class_indices
print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)


























