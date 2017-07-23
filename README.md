# deeplearn-caltech101

## Overview
Built a CNN to classify images in the Caltech-101 dataset. I used transfer learning instead of building the CNN from scratch. I built the 
model on top of the VGG19 model trained on Imagenet. I replaced the fully connected layers of the VGG19 and used the model to classify 
the Caltech Images. I did not fine tune the ConvNet because the Caltech-101 dataset is much smaller than the Imagenet dataset and the data
is really similar; also, I did not want to overfit to the data. 

## Hardware
Used a Google Compute Engine instance with 24 vCPUs and 24GB RAM. It took about 5.5 hours to finish. I used tensorflow as the backend 
for keras and built tensorflow using Intel MKL for optimization. 

## Results

#### Accuracy: ~95%

Epoch 1/20
6073/6073 [==============================] - 952s - loss: 3.1713 - acc: 0.3514

Epoch 2/20
6073/6073 [==============================] - 950s - loss: 1.9371 - acc: 0.5376

Epoch 3/20
6073/6073 [==============================] - 936s - loss: 1.4464 - acc: 0.6280

Epoch 4/20
6073/6073 [==============================] - 926s - loss: 1.1186 - acc: 0.7005

Epoch 5/20
6073/6073 [==============================] - 933s - loss: 0.9154 - acc: 0.7435

Epoch 6/20
6073/6073 [==============================] - 935s - loss: 0.7437 - acc: 0.7881

Epoch 7/20
6073/6073 [==============================] - 935s - loss: 0.6504 - acc: 0.8131

Epoch 8/20
6073/6073 [==============================] - 936s - loss: 0.5573 - acc: 0.8449

Epoch 9/20
6073/6073 [==============================] - 921s - loss: 0.4732 - acc: 0.8645

Epoch 10/20
6073/6073 [==============================] - 914s - loss: 0.4100 - acc: 0.8816

Epoch 11/20
6073/6073 [==============================] - 937s - loss: 0.4081 - acc: 0.8954

Epoch 12/20
6073/6073 [==============================] - 939s - loss: 0.3531 - acc: 0.9005

Epoch 13/20
6073/6073 [==============================] - 936s - loss: 0.3163 - acc: 0.9129

Epoch 14/20
6073/6073 [==============================] - 932s - loss: 0.3000 - acc: 0.9205

Epoch 15/20
6073/6073 [==============================] - 914s - loss: 0.2821 - acc: 0.9261

Epoch 16/20
6073/6073 [==============================] - 926s - loss: 0.2518 - acc: 0.9331

Epoch 17/20
6073/6073 [==============================] - 960s - loss: 0.2424 - acc: 0.9368

Epoch 18/20
6073/6073 [==============================] - 944s - loss: 0.2346 - acc: 0.9422

Epoch 19/20
6073/6073 [==============================] - 928s - loss: 0.2496 - acc: 0.9412

Epoch 20/20
6073/6073 [==============================] - 909s - loss: 0.2041 - acc: 0.9488
