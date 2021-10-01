# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import mediapipe as mp
import dlib
import cv2
import os

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=10, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                            self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                    self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def cropFace(self, img):
        h, w, ch = img.shape
        img, faces = self.findFaceMesh(img, draw=False)
        faces[0] = np.array(faces[0])

        # extract jawline
        jawpoints = np.array([132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361])
        jawline = faces[0][jawpoints]
        top = min(jawline[:, 1])
        bottom = max(jawline[:, 1])

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
        
        return result

if __name__ == '__main__':
    img = cv2.imread('uploads/thuy.JPG')
    detector = FaceMeshDetector()
    result = detector.cropFace(img)
    cv2.imshow('masked image', result)
    cv2.waitKey(0)

    