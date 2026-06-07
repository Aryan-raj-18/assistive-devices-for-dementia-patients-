
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")


url = 'http://192.168.3.119/cam-hi.jpg'

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)
            cv2.imshow('live transmission', im)
        except Exception as e:
            print("Error in run1:", e)

        if cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyWindow('live transmission')

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            bbox, label, conf = cv.detect_common_objects(im)
            im = draw_bbox(im, bbox, label, conf)
            cv2.imshow('detection', im)
        except Exception as e:
            print("Error in run2:", e)

        if cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyWindow('detection')


if __name__ == '__main__':
    print("started")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(run1)
        executor.submit(run2)
