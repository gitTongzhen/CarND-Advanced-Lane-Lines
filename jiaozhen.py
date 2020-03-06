import pickle
import cv2

img = cv2.imread('./test_images/straight_lines1.jpg')


g = open('calibration.p', 'rb')
result = pickle.load(g)
mtx = result["mtx"]
dist = result["dist"]

def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

dst = undistort(img)

