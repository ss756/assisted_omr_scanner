import cv2
import numpy as np
import nms


def squareContours(image, template, dim, qout):
    h_template = dim[0]["height"]
    w_template = dim[0]["width"]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    square_cnts = []
    for pt in zip(*loc[::-1]):
        square_cnts.append((pt[0], pt[1], pt[0] +
                           w_template, pt[1] + h_template))
    square_cnts = np.array(square_cnts)
    print("[x] The number of square contours detected is ", len(square_cnts))
    pick = nms.non_max_suppression_fast(square_cnts, 0.4)
    pick = sorted(pick, key=lambda ctr: ctr[0] + ctr[1] * 9)
    print("[x] After applying non-max-suppression, %d bounding boxes" % (len(pick)))
    qout.put(pick)
