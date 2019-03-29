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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#z-score
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
dropout_rate = 0.2
l = 12

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
  model.summary()


  # determine Loss function and Optimizer
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

  return model

# fix random seed for reproducibility
seed = 7

estimator = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=100, verbose=False)
#kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_train, y_train)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print(accuracy_score(y_test_or, prediction))
