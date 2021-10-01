from imutils import face_utils
import imutils
import numpy as np
import collections
import dlib
import cv2

def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image

def crop_face_using_landmarks(image):

    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    out_face = np.zeros_like(image)
    predictor_model = "./model/shape_predictor_81_face_landmarks.dat"

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    results = []

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #initialize mask array
        remapped_shape = np.zeros_like(shape) 
        feature_mask = np.zeros((image.shape[0], image.shape[1]))   

        # we extract the face
        remapped_shape = face_remap(shape)
        cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
        feature_mask = feature_mask.astype(np.bool)
        out_face[feature_mask] = image[feature_mask]
        results.append(out_face)

    return results
