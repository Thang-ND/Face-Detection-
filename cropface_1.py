'''
OPTION 1: CROP FACE USING 81 LANDMARKS 

Run on terminal: python option_1.py -i abc.jpg
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
    name, ext = fname.split('.')

    # STEP 1: REMOVE BACKGROUND FROM ORIGINAL IMAGE
    image = remove_background(path_img)

    # STEP 2: CROP FACE REGIONS
    images = crop_rectangle(image)

    # STEP 3: CROP FACE EXACTLY BY LANDMARKS
    for (i, img) in enumerate(images):
        fname = '{}_{}.{}'.format(name, i, ext)
        fname = os.path.join('result', fname)
        results = crop_face_using_landmarks(img)
        for result in results:
            cv2.imwrite(fname, result)
            # cv2.imshow("Image", result)
            # cv2.waitKey(0)
    
    print(fname)


if __name__ == '__main__':
    main()