from model_builder import ModelBuilder
import time
import cv2
import sys
import numpy as np
import cv2.aruco as aruco
from imutils import perspective
from pynput.keyboard import Key, Controller


# read command line parameters
if len(sys.argv) >= 1:
    path = sys.argv[1]
    try:
        video_id = int(sys.argv[2])
    except:
        video_id = 0
else:
    print("ERROR: Please specify path to HaGRID dataset")
    sys.exit(1)


IMG_SIZE = 64
COLOR_CHANNELS = 1
PATH = path
GESTURES = ['like', 'stop', 'dislike']


print("preparing model...")
model_builder = ModelBuilder(COLOR_CHANNELS, IMG_SIZE, PATH, GESTURES)
# -----------------------------------------------------
# no model yet? prepare and save for later (if you want)
model_builder.prepare_model()
model_builder.save_model()
# -----------------------------------------------------
# model already existing? just load it
# model_builder.load_model()
# -----------------------------------------------------
print("model prep done")


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()


keyboard = Controller()
cap = cv2.VideoCapture(video_id)
bounding_box = []


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    # if aruco board is moved and again 4 corners are detected, set new bounding box
    if corners and len(corners) == 4:
        boundings = []
        for i, corner in enumerate(corners):
            boundings.append([corner[0][0][0], corner[0][0][1]])
        bounding_box = perspective.order_points((np.array(boundings)))

    # if there is a bounding box, gesture recognition can start
    if len(bounding_box) == 4:
        # warp frame to bounding box for hand
        destination = np.float32(np.array([[0, 0], [IMG_SIZE, 0], [IMG_SIZE, IMG_SIZE], [0, IMG_SIZE]]))
        matrix = cv2.getPerspectiveTransform(bounding_box, destination)
        to_warp = frame if COLOR_CHANNELS == 3 else gray
        warped = cv2.warpPerspective(to_warp, matrix, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)

        # define X_test and predict
        X_test = warped.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
        prediction = model_builder.predict(X_test)
        gesture = model_builder.label_names[np.argmax(prediction)]

        # handle recognized gesture
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
    else:
        print('No bounding box found yet.')
    time.sleep(0.5)

