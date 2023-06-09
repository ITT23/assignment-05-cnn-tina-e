import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import random

# import a lot of things from keras:
# sequential model
from keras.models import Sequential

# layers
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomFlip, RandomRotation, RandomContrast, RandomBrightness

# loss function
from keras.metrics import categorical_crossentropy

# callback functions
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# convert data to categorial vector representation
from keras.utils import to_categorical

# nice progress bar for loading data
from tqdm.notebook import tqdm

# helper function for train/test split
from sklearn.model_selection import train_test_split

# import confusion matrix helper function
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import pre-trained model
from keras.applications.vgg16 import VGG16

# include only those gestures
CONDITIONS = ['like', 'stop', 'dislike']

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)

# number of color channels we want to use
# set to 1 to convert to grayscale
# set to 3 to use color images
COLOR_CHANNELS = 1


class ModelBuilder:
    def __init__(self):
        self.annotations = dict()
        self.images = []
        self.labels = []
        self.label_names = []
        self.X_train = self.X_test = self.y_train = self.y_test = self.train_label = self.test_label = []
        self.reduce_lr = self.stop_early = self.model = None
        self.batch_size = 8
        self.epochs = 10
        self.activation = 'relu'
        self.activation_conv = 'LeakyReLU'  # LeakyReLU
        self.layer_count = 2
        self.num_neurons = 64

    def load_annotations(self):
        for condition in CONDITIONS:
            with open(f'D:/Week6/gesture_dataset_sample/_annotations/{condition}.json') as f:
                self.annotations[condition] = json.load(f)

    def preprocess_image(self, img):
        if COLOR_CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img, SIZE)
        return img_resized

    def load_images_with_annotations(self):
        for condition in CONDITIONS:
            for filename in os.listdir(f'D:/Week6/gesture_dataset_sample/{condition}'):
                # extract unique ID from file name
                UID = filename.split('.')[0]
                img = cv2.imread(f'D:/Week6/gesture_dataset_sample/{condition}/{filename}')

                # get annotation from the dict we loaded earlier
                try:
                    annotation = self.annotations[condition][UID]
                except Exception as e:
                    print(e)
                    continue

                # iterate over all hands annotated in the image
                for i, bbox in enumerate(annotation['bboxes']):
                    # annotated bounding boxes are in the range from 0 to 1
                    # therefore we have to scale them to the image size
                    x1 = int(bbox[0] * img.shape[1])
                    y1 = int(bbox[1] * img.shape[0])
                    w = int(bbox[2] * img.shape[1])
                    h = int(bbox[3] * img.shape[0])
                    x2 = x1 + w
                    y2 = y1 + h

                    # crop image to the bounding box and apply pre-processing
                    crop = img[y1:y2, x1:x2]
                    preprocessed = self.preprocess_image(crop)

                    # get the annotated hand's label
                    # if we have not seen this label yet, add it to the list of labels
                    label = annotation['labels'][i]
                    if label not in self.label_names:
                        self.label_names.append(label)

                    label_index = self.label_names.index(label)

                    self.images.append(preprocessed)
                    self.labels.append(label_index)

    def define_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
        self.X_train = np.array(self.X_train).astype('float32')
        self.X_train = self.X_train / 255.

        self.X_test = np.array(self.X_test).astype('float32')
        self.X_test = self.X_test / 255.

        y_train_one_hot = to_categorical(self.y_train)
        y_test_one_hot = to_categorical(self.y_test)

        self.train_label = y_train_one_hot
        self.test_label = y_test_one_hot

        self.X_train = self.X_train.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
        self.X_test = self.X_test.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

    def build_model(self):
        num_classes = len(self.label_names)

        # define model structure
        # with keras, we can use a model's add() function to add layers to the network one by one
        model = Sequential()

        # data augmentation (this can also be done beforehand - but don't augment the test dataset!)
        model.add(RandomFlip('horizontal'))
        model.add(RandomContrast(0.1))
        # model.add(RandomBrightness(0.1))
        # model.add(RandomRotation(0.2))

        # first, we add some convolution layers followed by max pooling
        model.add(
            Conv2D(64, kernel_size=(9, 9), activation=self.activation_conv, input_shape=(SIZE[0], SIZE[1], COLOR_CHANNELS),
                   padding='same'))
        model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

        model.add(Conv2D(32, (5, 5), activation=self.activation_conv, padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

        model.add(Conv2D(32, (3, 3), activation=self.activation_conv, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        # dropout layers can drop part of the data during each epoch - this prevents overfitting
        model.add(Dropout(0.2))

        # after the convolution layers, we have to flatten the data so it can be fed into fully connected layers
        model.add(Flatten())

        # add some fully connected layers ("Dense")
        for i in range(self.layer_count - 1):
            model.add(Dense(self.num_neurons, activation=self.activation))

        model.add(Dense(self.num_neurons, activation=self.activation))

        # for classification, the last layer has to use the softmax activation function, which gives us probabilities for each category
        model.add(Dense(num_classes, activation='softmax'))

        # specify loss function, optimizer and evaluation metrics
        # for classification, categorial crossentropy is used as a loss function
        # use the adam optimizer unless you have a good reason not to
        model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

        # define callback functions that react to the model's behavior during training
        # in this example, we reduce the learning rate once we get stuck and early stopping
        # to cancel the training if there are no improvements for a certain amount of epochs
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
        self.stop_early = EarlyStopping(monitor='val_loss', patience=3)
        self.model = model

    def fit_model(self):
        history = self.model.fit(
            self.X_train,
            self.train_label,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.X_test, self.test_label),
            callbacks=[self.reduce_lr, self.stop_early]
        )
        return history

    def display_summary(self):
        print(self.model.summary())

    def prepare_model(self):
        self.load_annotations()
        self.load_images_with_annotations()
        self.define_train_test()
        self.build_model()
        hist = self.fit_model()
        print(hist)
        self.display_summary()

    def predict(self, input):
        prediction = self.model.predict(input)
        print(self.label_names[np.argmax(prediction)], np.max(prediction))
