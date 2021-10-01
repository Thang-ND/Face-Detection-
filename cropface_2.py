'''
OPTION 2: CROP FACE USING MODNET TO REMOVE BACKGROUND, THEN CROP BY JAW MARKS

Run on terminal: python option_2.py -i abc.jpg
'''
import cv2
import argparse
import os, sys
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='input_path', type=str, help='path of input images')
args = parser.parse_args()
# check input arguments
path_img = args.input_path
if not os.path.exists(path_img):
    print('Cannot find input path: {0}'.format(path_img))
    exit()

def main():
    
    fname = path_img.split('/')[-1]
    name = fname.split('.')[0]
    ext = 'png'

    # STEP 1: REMOVE BACKGROUND FROM ORIGINAL IMAGE
    image = remove_background(path_img)

    # STEP 2: CROP FACE EXACTLY BY JAW
    fname = '{}.{}'.format(name, ext)
    fname = os.path.join('result', fname)
    detector = FaceMeshDetector()
    result = detector.cropFace(image)
    cv2.imwrite(fname, result)
    cv2.imshow("Image", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    main() 