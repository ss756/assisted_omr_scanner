import numpy as np
import cv2


def meta_data(bb1, bb2):
    # boundary contour
    x_start, y_start, x_end, y_end = bb2
    # print("Print bb1 is ", bb1)
    # print("Print bb2 is", bb2)
    x0, y0, x1, y1 = bb1
    # print(f"{x0} start {x1} end {y0} start {y1} end")
    xc = int((x0 + x1)//2)
    yc = int((y0 + y1)//2)
    if(x_start < xc and xc < x_end and y_start < yc and yc < y_end):
        return True
    else:
        return False

def check_bboxes(img, boundary_abs, qout):
    '''
    Refer - > https://stackoverflow.com/questions/62801070/detect-checkboxes-from-a-form-using-opencv-python
    :param img:
    :param boundary_abs:
    :return:
    '''
    img_height, img_width, _ = img.shape
    # print("[i] The area of the image is ", img_height*img_width)
    thresh_y = int(33 * img_height / 1152)
    thresh_x = int(33 * img_width / 864)
    x0 = int((boundary_abs[0] * img_width) - thresh_x)
    y0 = int((boundary_abs[1] * img_width) - thresh_y)
    x1 = int((boundary_abs[2] * img_width) + thresh_x)
    y1 = int((boundary_abs[3] * img_height) + thresh_y)
    bb2 = (x0, y0, x1, y1)

    # TO VISUALIZE THE NAME SECTION UNCOMMENT THE BELOW THREE LINES
    # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 30)
    # cv2.imshow("Checkbox Image", cv2.resize(img, (720, 900)))
    # cv2.waitKey(0)


    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
    img_width, img_height, _ = img.shape
    lineWidth = 7
    lineMinWidth = 7
    cnts = []
    # kernal1 = np.ones((lineWidth, lineWidth), np.uint8)
    kernal1h = np.ones((1, lineWidth), np.uint8)
    kernal1v = np.ones((lineWidth, 1), np.uint8)

    # kernal6 = np.ones((lineMinWidth, lineMinWidth), np.uint8)
    kernal6h = np.ones((1, lineMinWidth), np.uint8)
    kernal6v = np.ones((lineMinWidth, 1), np.uint8)

    # detect horizontal line
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1h) # bridge small gap in horizonntal lines
    img_bin_h = cv2.morphologyEx(img_bin_h, cv2.MORPH_OPEN, kernal6h) # kep ony horiz lines by eroding everything else in hor direction

    ## detect vert lines
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)  # bridge small gap in vert lines
    img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)# kep ony vert lines by eroding everything else in vert direction

    ### function to fix image as binary
    def fix(img):
        img[img > 127] = 255
        img[img < 127] = 0
        return img
    img_bin_final = fix(fix(img_bin_h) | fix(img_bin_v))
    finalKernel = np.ones((5, 5), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, finalKernel, iterations=1)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    ### skipping first two stats as background

    for x, y, w, h, area in stats[2:]:
        bb1 = (x, y, x+w, y+h)
        ar = (bb1[2] - bb1[0])/(bb1[3]-bb1[1])
        rel_area = area/(img_height*img_width)
        if meta_data(bb1, bb2) and rel_area > 0.00010063905801841694 and rel_area < 0.005031952900920848 and ar < 1.1:
            cnts.append((x, y, x+w, y+h))

    qout.put(cnts)


