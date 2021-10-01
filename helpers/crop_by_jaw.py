# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os


def crop_by_jaw(img):
    predictor_model = "./model/shape_predictor_81_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # load image
    h, w, ch = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    if ch < 4:
        b,g,r = cv2.split(img)
        a = np.ones((h,w,1), np.uint8) * 255
        img = cv2.merge((b, g, r, a))

    # detect face
    rects = detector(gray, 1)

    results = []

    for (i, rect) in enumerate(rects):
        """
        Determine the facial landmarks for the face region, then convert the facial landmark (x, y) - coordinates to a NumPy array
        """
        #roi = rects[0] # region of interest
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract jawline
        jawline = shape[3:14]
        top = min(jawline[:,1])
        bottom = max(jawline[:,1])

        # extend contour for masking
        jawline = np.append(jawline, [ w-1, jawline[-1][1] ]).reshape(-1, 2)
        jawline = np.append(jawline, [ w-1, h-1 ]).reshape(-1, 2)
        jawline = np.append(jawline, [ 0, h-1 ]).reshape(-1, 2)
        jawline = np.append(jawline, [ 0, jawline[0][1] ]).reshape(-1, 2)
        contours = [ jawline ]
        
        # generate mask
        mask = np.ones((h,w,1), np.uint8) * 255 # times 255 to make mask 'showable'
        cv2.drawContours(mask, contours, -1, 0, -1) # remove below jawline
        
        # apply to image
        result = cv2.bitwise_and(img, img, mask = mask)
        
        # result = result[top:bottom, roi.left():roi.left()+roi.width()] # crop ROI
        #cv2.imwrite('result.png', result)
        # cv2.imshow('masked image', result)
        # cv2.waitKey(0)
        results.append(result)

    return results

# if __name__ == '__main__':
#     im = cv2.imread('hanglinh.jpeg')
#     results = crop_by_jaw(im)
#     for result in results:
#         cv2.imshow('masked image', result)
#         cv2.waitKey(0)

    