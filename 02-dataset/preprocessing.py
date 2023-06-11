import json
import cv2
import uuid
import os
import sys


WIDTH = 1080
HEIGHT = 1920
WINDOW_NAME = 'select bounding box'
images = []


# read images and resize to 1080x1920
path_original = 'images_original'
path_resized = 'images_resized'
for img_path in os.listdir(path_original):
   img = cv2.imread(f'{path_original}/{img_path}')
   resized = cv2.resize(img, (WIDTH, HEIGHT))
   # save image data with label
   images.append((img_path.split('-')[0], resized))


counter = 0
corners = []
annotations = dict()

label = images[counter][0]
img_original = images[counter][1]
img = img_original.copy()


def save():
    with open('annot_emmert.json', 'w') as fp:
        json.dump(annotations, fp)
    sys.exit(0)


def add_corner(x: int, y: int):
    """
    select bounding box by clicking top left and then bottom right corner
    add corner on click, save annotations on 2nd corner clicked
    :param x: x corner
    :param y: y corner
    """
    global img, img_original, images, counter, label, annotations, corners, WINDOW_NAME, WIDTH, HEIGHT
    corners.append([x, y])
    if len(corners) == 2:
        x_top = corners[0][0]
        y_top = corners[0][1]
        x_bottom = corners[1][0]
        y_bottom = corners[1][1]

        x_top_rel = x_top / WIDTH
        y_top_rel = y_top / HEIGHT
        width_rel = (x_bottom - x_top) / WIDTH
        height_rel = (y_bottom - y_top) / HEIGHT

        guid = str(uuid.uuid4())
        annotation = dict()
        annotation["bboxes"] = [[x_top_rel, y_top_rel, width_rel, height_rel]]
        annotation["labels"] = [label]
        annotations[guid] = annotation

        # write files to use them later for prediction
        cv2.imwrite(f'{path_resized}/{guid}.jpg', img_original)

        counter += 1
        # if all the images are annotated, save to annot.json
        if counter >= len(images):
            save()
            return
        label = images[counter][0]
        img_original = images[counter][1]
        img = img_original.copy()
        corners = []
        cv2.imshow(WINDOW_NAME, img)
    else:
        img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(WINDOW_NAME, img)


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 2:
        add_corner(x, y)


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback, (img))
cv2.imshow(WINDOW_NAME, img)


if __name__ == '__main__':
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            sys.exit(0)

   

   

