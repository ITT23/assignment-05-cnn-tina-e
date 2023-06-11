[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/P2j0joSQ)

# 01 Exploring Image Resolution
Notebook describes the exploration of different image resolutions (12x12 - 256x256) on training.

---

# 02 Dataset
- gesture_recognition_own: Pretrained model if one do not want to train again :)
- images_original: Folder with images in original resolution
- images_resized: Folger with images in 1080x1920 and uuid file names (gained from 'preprocessing.py')
- preprocessing.py: Resizes images to 1080x1920 and allows to annotate images (bounding boxes) by clicking at the top left corner and bottom right corner of the wished bounding box; writes the annotation json file based on these
- predict.ipybn: Trains a model for *like, dislike, rock, peace, stop* gestures + predictions for own images

**! You might adjust the PATH to the HaGRID dataset in the notebook**

---

# 03 Media Control
- gesture_recognition_media: Pretrained model if one do not want to train again :)
- **media_control.py: run this file to start**
  - **command line params: 1) path to HaGRID dataset (required) and 2) video input device (optional, default: 0)**
  - **how to: perform gestures with an aruco board as background**
  - **like gesture: volume up**
  - **dislike gesture: volume down**
  - **stop gesture: play/pause**
- model_builder.py: ModelBuilder handles training and prediction

