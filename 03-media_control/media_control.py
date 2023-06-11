#%%

from model_builder import ModelBuilder
from enum import Enum
import time
import cv2
import sys
import numpy as np
import cv2.aruco as aruco
from imutils import perspective
from tensorflow import keras
import tensorflow as tf
from pynput.keyboard import Key, Controller
#%%

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%

class Gesture(Enum):
    START_PAUSE = 0
    VOL_UP = 1
    VOL_DOWN = 2


GESTURE_LABELS = ['stop', 'like', 'dislike']
#lables = ['like', 'no_gesture', 'stop', 'dislike']

IMG_SIZE = 64
COLOR_CHANNELS = 3 # TODO: angleichen


model_builder = ModelBuilder()
model_builder.prepare_model()
#new_model = keras.models.load_model('../gesture_recognition')

print("model prep done")

#%%

# read webcam
video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

keyboard = Controller()

cap = cv2.VideoCapture(video_id)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    print(len(corners))
    boundings = []
    if corners and len(corners) == 4:
        for i, corner in enumerate(corners):
            boundings.append([corner[0][0][0], corner[0][0][1]])
        bounding_box = perspective.order_points((np.array(boundings)))

        destination = np.float32(np.array([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]]))
        matrix = cv2.getPerspectiveTransform(bounding_box, destination)
        warped = cv2.warpPerspective(gray, matrix, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)

        #resized = cv2.resize(warped, IMG_SIZE)

        reshaped = warped.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        # cv2.imshow("test", frame)
        cv2.imwrite(f'{time.time()}.jpg', warped)

        prediction = model_builder.predict(reshaped)
        print(prediction)
        # print(np.argmax(prediction))
        print(model_builder.labels[np.argmax(prediction)], np.max(prediction))

        # TODO: check why not reaching statements
        # TODO: save bounding box if no one found
        gesture = model_builder.labels[np.argmax(prediction)]
        if gesture == 'stop':
            print('play pause')
            keyboard.press(Key.media_play_pause)
            keyboard.release(Key.media_play_pause)
        elif gesture == 'like':
            print('like')
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
        elif gesture == 'dislike':
            print('dislike')
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
    time.sleep(0.5)

