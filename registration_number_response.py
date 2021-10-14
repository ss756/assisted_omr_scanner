import config
import os
import cv2
import numpy as np


reg_number_metadata = [['G', 'D', 'S'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                       ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                       ['A', 'C', 'N', 'T', 'S'], ['F', 'L']]

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
    if data[0] < seq_data[0] and (abs(seq_data[0] - data[0]) + abs(seq_data[2] - data[2]) > abs(seq_data[1] - data[1]) + abs(seq_data[3] - data[3])):
        return True
    else:
        return False

def getData(image):
    image_copy = image.copy()
    height, width, _ = image.shape
    points = config.points
    channels, rows, columns = points.shape
    for row in range(rows):
        for column in range(columns):
            if column == 0:
                points[:, row, column] = points[:, row, column] * width
            else:
                points[:, row, column] = points[:, row, column] * height
    points = points.astype(int)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    # cv2.imshow("cropped", cv2.resize(cropped, (720,900)))
    # cv2.waitKey(0)
    img = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        ar = w / h
        if area > 250 and ar < 1.1:
            final_cnts.append((x, y, x+w, y+h))
    final_cnts = sorted(final_cnts, key=lambda bbox: bbox[0] + bbox[1] * 9, reverse=False)

    for cnt in final_cnts:
        x, y, x1, y1 = cnt
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 7)
    # cv2.imshow("Image", cv2.resize(img, (720, 900)))
    # cv2.waitKey(0)


    total_cnts = len(final_cnts)
    bbox_idx = 0
    col_idx = 9
    bbox_data = [None for i in range(total_cnts)]
    reg_no_bbubled_bboxes = [[] for j in range(col_idx + 1)]
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
        reg_no_bbubled_bboxes[index_position].append(final_cnts[start])
        start += 1
    final_code = ''
    for col_idx in range(len(reg_no_bbubled_bboxes)):
        im_copy = img_copy.copy()
        whitePixels = []
        for idx in range(len(reg_no_bbubled_bboxes[col_idx])):
            x, y, x1, y1 = reg_no_bbubled_bboxes[col_idx][idx]
            cv2.rectangle(im_copy, (x, y), (x1, y1), (255, 0, 0), 3)
            number_of_white_pix = np.sum(img_copy[y:y1, x:x1] == 255)
            whitePixels.append(number_of_white_pix)
        min_value = min(whitePixels)
        if min_value > 5000: # set a threshold
            min_index = whitePixels.index(min_value)
            x, y, x1, y1 = reg_no_bbubled_bboxes[col_idx][min_index]
            cv2.rectangle(im_copy, (x, y), (x1, y1), (0, 255, 0), 3)
            final_code += reg_number_metadata[col_idx][min_index]
        else:
            print("Contour is left blank!!")
        # print("White pixels debug", whitePixels)
        # cv2.imshow("Finale", cv2.resize(im_copy, (720,900)))
        # cv2.waitKey(0)
    # print("Final Code is", final_code)

    return final_code