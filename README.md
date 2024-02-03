# Minial Application for Athlete Tracking

This application implements a naive tracking algorithm based on the YOLOv8 detector. The application loads a video file and processes each frame. On each frame the YOLOv8 detector is used for detecting persons. The detection are drawn on each frame and saved to an output video, additionally the detected bounding boxes are also saved in a text file. Optionally, the trajectory is also drawn.

The application is implemented in C++ using the OpenCV library for loading the YOLOv8 model, loading videos, drawing on images and saving videos.

![alt-text](drill_1_out.gif)