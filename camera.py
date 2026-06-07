import urllib.request
import cv2
import numpy as np

url = 'http://192.168.3.119'

try:
    img_resp = urllib.request.urlopen(url, timeout=5)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)
    cv2.imshow("Test", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print("Camera error:", e)
