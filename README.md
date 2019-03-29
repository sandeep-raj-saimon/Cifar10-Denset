# Cifar10-Denset

densenet.py consists of a densenet architecture without a dense layer as final classifier. Instead we are using Kerasclassifier to make the classifier.
The feature extracted from the model is fed into the KerasClassifier and is used to train the classifier.
Tensorflow has been used for image augmentaion
