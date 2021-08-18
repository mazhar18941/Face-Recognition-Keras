# import the necessary packages
import numpy as np
import cv2

#defining prototext and caffemodel paths


def load_model(prototextPath, caffeModel):
    return cv2.dnn.readNet(prototextPath, caffeModel)

def get_detection(face_detector, image):

    (h,w) = image.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()
    return detections


def get_face_pixels(image, detections, threshold=0.5):
    # loop over the detections
    (h,w) = image.shape[:2]
    bb_pixels = []
    bb_pixels_coord = []
    for i in range(0, detections.shape[2]):
     # extract the confidence and prediction

        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            pixels = image[startY:endY, startX:endX, :]
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            bb_pixels.append(pixels)
            bb_pixels_coord.append([startX, startY, endX, endY])
            
    return bb_pixels_coord, bb_pixels