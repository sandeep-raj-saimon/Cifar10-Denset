import keras
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import regularizers
from sklearn.metrics import accuracy_score
import numpy as np

# for merging keras with scikit-learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold

# this part will prevent tensorflow to allocate all the avaliable GPU Memory
# backend
import tensorflow as tf
from keras import backend as  k                                                             

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 1
l = 40
num_filter = 12
compression = 0.5
dropout_rate = 0.0
IMAGE_SIZE = 32
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

weight_decay = 1e-4

# Dense Block
def add_denseblock(input, num_filter = 12, dropout_rate = 0.0):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=regularizers.l2(weight_decay))(relu)
        if dropout_rate>0:
          Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp
	
	
def add_transition(input, num_filter = 12, dropout_rate = 0.0):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=regularizers.l2(weight_decay))(relu)
    if dropout_rate>0:
      Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg
	
def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    conv_output1 = Conv2D(10, (2,2), use_bias=False ,padding='same',kernel_regularizer=regularizers.l2(weight_decay))(AvgPooling)
    conv_output = Conv2D(10, (2,2), use_bias=False ,padding='valid',kernel_regularizer=regularizers.l2(weight_decay))(conv_output1)

    print(AvgPooling.shape)
    print(conv_output1.shape)
    print(conv_output.shape)

    flat = Flatten()(conv_output)
    print(flat.shape)
    #output = Dense(num_classes, activation='softmax')(flat)
    
    return flat
	
num_filter = 12
dropout_rate = 0.0
l = 12

# IMAGE AUGMENTATION bEGINS

#rotation

from math import pi

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images)
    
    
    X = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate
	
# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
rotated_imgs = rotate_images(x_train[:1000], -90, 90, 10)
x_train = np.concatenate((x_train, rotated_imgs), axis=0)
print("rotated")
repeat = np.repeat(y_train[:1000], 10)  
repeat = repeat.reshape(10000,1)
y_train = np.concatenate((y_train,repeat),axis=0)

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([32, 32], dtype = np.int32)
    
    X_scale_data = []
    X = tf.placeholder(tf.float32, shape = (1,32, 32, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)# on each image 3 operations are perfomred
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

scaled_imgs = central_scale_images(x_train[1000:3000], [0.90, 0.75, 0.60])
x_train = np.concatenate((x_train, scaled_imgs), axis=0)
print("central scale")
print(x_train.shape)
repeat = np.repeat(y_train[1000:3000], 3)  
repeat = repeat.reshape(6000,1)
y_train = np.concatenate((y_train,repeat),axis=0)
print(x_train.shape)
print(y_train.shape)

# translation of the image
from math import ceil, floor

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE)) 
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE 
        
    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4# each image is translated 4 times
    X_translated_arr = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3), 
				    dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset 
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            
            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
			 w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr
	
translated_imgs = translate_images(x_train[3000:5000])
x_train = np.concatenate((x_train, translated_imgs), axis=0)
print("translate")
print(x_train.shape)
repeat = np.repeat(y_train[3000:5000], 4)  
repeat = repeat.reshape(8000,1)
y_train = np.concatenate((y_train,repeat),axis=0)
print(y_train.shape)

# flipping of images
def flip_images(X_imgs):
    X_flip = []
    
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})# on each image 3 operations are performed
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip
	
flipped_images = flip_images(x_train[6000:9000])
x_train = np.concatenate((x_train, flipped_images), axis=0)
print("flip")
print(x_train.shape)
repeat = np.repeat(y_train[6000:9000], 3)  
repeat = repeat.reshape(9000,1)
y_train = np.concatenate((y_train,repeat),axis=0)

# IMAGE AUGMENTATION ENDS


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#z-score (for normalization)
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 10

# one-hot encoding
y_train_or = y_train
y_train = np_utils.to_categorical(y_train,num_classes)
y_test_or = y_test
y_test = np_utils.to_categorical(y_test,num_classes)

# Architecture of the model DenseNet
def create_model():
  
  input = Input(shape=(img_height, img_width, channel,))
  First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)

  First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
  First_Transition = add_transition(First_Block, num_filter, dropout_rate)

  Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
  Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

  Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
  Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

  Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
  output = output_layer(Last_Block)


  model = Model(inputs=[input], outputs=[output])
#  model.summary()

  # determine Loss function and Optimizer
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

  return model

estimator = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=100, verbose=False)
#kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_train, y_train)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print(accuracy_score(y_test_or, prediction))

"""
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))"""
