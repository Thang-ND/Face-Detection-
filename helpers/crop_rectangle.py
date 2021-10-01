import imutils
import dlib
import cv2

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def crop_rectangle(image):
    # load the input image, resize it, and convert it to grayscale
    height, width, channel = image.shape
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    images = []

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        (x, y, w, h) = rect_to_bb(rect)
        deltaX = int(0.15 * w)
        deltaY = int(0.15 * h)
        Xmin = x - deltaX if x - deltaX > 0 else 0
        Xmax = x + w + deltaX if x + w + deltaX < width else width
        Ymin = y - 5 * deltaY if y - 5 * deltaY > 0 else 0
        Ymax = y + w + deltaY if y + w + deltaY < height else height

        roi = image[Ymin : Ymax, Xmin : Xmax]
        images.append(roi)

    return images