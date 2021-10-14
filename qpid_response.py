import os
import cv2
import numpy as np
import math
import imutils
from imutils.perspective import four_point_transform

qpid_metaData = [['A', 'B', 'C', 'D', 'E'], ['V', 'W', 'X', 'Y', 'Z'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ['1', '2', '3', '4', '5'] ]

def get_gradients(data, seq_data):
    '''

    ************************************************************************
     Rearrange word tokens based on their positions
    ************************************************************************

    :param: data -> x-y coordinates of ith token
    :param: seq_data -> x-y coordinates of the (i+1)th token
    :returns bool -> True if the token belongs to the same line else False

    '''

    # to lie in the same line x0 of the ith token should be less than x0 of the (i+1)th token
    # rate of change of x should be greater than the rate of change of y
    #
    if data[0] < seq_data[0] and (abs(seq_data[0] - data[0]) + abs(seq_data[2] - data[2]) > abs(seq_data[1] - data[1]) + abs(seq_data[3] - data[3])):
        return True  # same line
    else:
        return False

def extractQPID(img, cord, blur=False, erode=False):
    height, width, _ = img.shape
    x, y, x1, y1 = cord
    x = math.ceil(x * width)
    y = math.ceil(y * height) + 12
    x1 = math.ceil(x1 * width)
    y1 = math.ceil(y1 * height)
    img = img[y:y1, x:x1]
    # cv2.imshow("Original Image is", img)
    # cv2.waitKey(0)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(edged, kernel)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]

    final_cnts = []
    for k in cnts:
        (x, y, w, h) = cv2.boundingRect(k)
        area = w * h
        if area > 200:
            final_cnts.append((x, y, x+w, y+h))
    final_cnts = sorted(final_cnts, key=lambda bbox: bbox[0] + bbox[1]*9, reverse=False)
    total_cnts = len(final_cnts)
    bbox_idx = 0
    col_idx = 4
    bbox_data = [None for i in range(len(final_cnts))]
    qpid_bbubled_bboxes = [[] for j in range(col_idx + 1)]
    while(bbox_idx < total_cnts - 1):
        data = final_cnts[bbox_idx]
        seq_data = final_cnts[bbox_idx+1]
        if get_gradients(data, seq_data):
            bbox_data[bbox_idx] = col_idx
        else:
            bbox_data[bbox_idx] = col_idx
            col_idx -= 1
        bbox_idx += 1
    bbox_data[-1] = col_idx
    start = 0
    end = len(final_cnts)
    while(start < end):
        index_position = bbox_data[start]
        qpid_bbubled_bboxes[index_position].append(final_cnts[start])
        start+=1

    # print("Final qpid response is",qpid_bbubled_bboxes)

    final_code = ''
    for col_idx in range(len(qpid_bbubled_bboxes)):
        im_copy = img_copy.copy()
        whitePixels = []
        for idx in range(len(qpid_bbubled_bboxes[col_idx])):
            x, y, x1, y1 = qpid_bbubled_bboxes[col_idx][idx]
            mask = np.zeros((h, w), dtype='uint8')
            total = cv2.countNonZero(mask)
            cv2.rectangle(im_copy, (x, y), (x1, y1), (255, 0, 0), 3)
            number_of_white_pix = np.sum(img_copy[y:y1, x:x1] == 255)
            whitePixels.append(number_of_white_pix)
        min_value = min(whitePixels)
        if min_value > 5000: # set a threshold
            min_index = whitePixels.index(min_value)
            x, y, x1, y1 = qpid_bbubled_bboxes[col_idx][min_index]
            cv2.rectangle(im_copy, (x, y), (x1, y1), (0, 255, 0), 3)
            final_code += qpid_metaData[col_idx][min_index]
        else:
            print("Contour is left blank!!")
        # print("White pixels debug", whitePixels)
        # cv2.imshow("Finale", im_copy)
        # cv2.waitKey(0)
    return final_code