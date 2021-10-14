import cv2
import config
from subprocess import *
import argparse
import numpy as np
import math
import os
from pathlib import Path


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
        return False  # the word token does not lie on the same line as the previous token


def meta_data_extraction(bb1, bb2):
    # boundary contour
    x_start, y_start, x_end, y_end = bb1
    x0, y0, x1, y1 = bb2
    xc = int((x0 + x1)//2)
    yc = int((y0 + y1)//2)
    if (x_start < xc and xc < x_end and y_start < yc and yc < y_end):
        return True
    else:
        return False


def main(bboxes, dims, q):
    final_data = []
    json_data = []
    img_width = dims[0]["width"]
    img_height = dims[0]["height"]
    # we do not want the candidate signature bbox
    for j in range(len(config.outlier_bboxes) - 1):
        bb1 = config.outlier_bboxes[config.names[str(j)]][0]
        # orig_img_height, orig_img_width
        x, y, x1, y1 = bb1
        x = int(x * img_width)
        y = int(y * img_height)
        x1 = int(x1 * img_width)
        y1 = int(y1 * img_height)
        bb1 = (x, y, x1, y1)
        for k in range(len(bboxes)):
            if meta_data_extraction(bb1, bboxes[k]):
                x_1, y_1, x1_1, y1_1 = bboxes[k]
                a = (x_1 / img_width)
                b = (y_1 / img_height)
                c = (x1_1 / img_width)
                d = (y1_1 / img_height)
                relevant_bbox = (a, b, c, d)
                json_data.append(
                    {
                        "type": j,
                        "word_tokens_loc": [],
                        "bbox_loc": relevant_bbox,
                    }
                )
                # json_data.append(
                #     {
                #         "type": j,
                #         "word_tokens_loc": [],
                #         "bbox_loc": bboxes[k],
                #     }
                # )
        final_data.append(json_data)
        json_data = []
    q.put(final_data)


def reg_words_bbox_extraction(sorted_data, q):
    line_width = 25  # make this relative
    digits_bbox = [[] for k in range(4)]
    bbox_idx = 0
    bbox_line_idx = 0
    line_data = [None for i in range(len(sorted_data))]
    while(bbox_idx < len(sorted_data) - 1):
        data = sorted_data[bbox_idx]
        seq_data = sorted_data[bbox_idx + 1]
        if get_gradients(data, seq_data):
            line_data[bbox_idx] = bbox_line_idx
        else:
            # print("[I] Found new line!")
            line_data[bbox_idx] = bbox_line_idx
            bbox_line_idx += 1
        bbox_idx += 1
    line_data[-1] = bbox_line_idx
    # print("[I] Line data is ", line_data)
    bbox_idx = 0
    idx = 0
    while(bbox_idx < len(sorted_data)):
        if line_data[bbox_idx] == idx:
            digits_bbox[idx].append(sorted_data[bbox_idx])
        else:
            digits_bbox[idx+1].append(sorted_data[bbox_idx])
            idx += 1
        bbox_idx += 1
    q.put(digits_bbox)

def storeImages(section_name, image_name, type, idx, image):
    root_path = os.getcwd()
    root_folder = f"{root_path}/tenant/bin/{type}"
    os.makedirs(root_folder, exist_ok=True)
    path = f"{root_folder}/{section_name}"
    file_path_folder = f"{path}/{image_name}"
    os.makedirs(file_path_folder, exist_ok=True)
    name = f"{file_path_folder}/{idx}.png"
    # print("[d] Writing Image to this path ", name)
    cv2.imwrite(f"{file_path_folder}/{idx}.png", image)








def reg_words_extraction(image_name,image, bboxes, qout):
    print("[x] Image name is", image_name)
    section_name = config.names[str(3)]
    line_width = 33     # make this relative
    height, width, _ = image.shape
    line_bboxes = []
    for line_idx in range(len(bboxes)):
        for bbox_idx in range(len(bboxes[line_idx])):
            x, y, x1, y1 = bboxes[line_idx][bbox_idx]
            x = math.ceil(width * x)
            y = math.ceil(height * y)
            x1 = math.floor(width * x1)
            y1 = math.floor(height * y1)
            cv2.rectangle(image, (x - 1, y - 1), (x1 + 5, y1 + 5),
                          (255, 255, 255), line_width)     # add relative thresholds
            img = image[y-1: y1+5, x-1:x1+5]
            storeImages(section_name, image_name.split(".")[-2], "individual_data", f"{line_idx}_{bbox_idx}", img)

        all_name_x0 = [bbox[0] for bbox in bboxes[line_idx]]
        all_name_y0 = [bbox[1] for bbox in bboxes[line_idx]]
        all_name_x1 = [bbox[2] for bbox in bboxes[line_idx]]
        all_name_y1 = [bbox[3] for bbox in bboxes[line_idx]]
        reg_name_bbox = [min(all_name_x0), min(all_name_y0),
                         max(all_name_x1), max(all_name_y1)]
        line_x0, line_y0, line_x1, line_y1 = reg_name_bbox
        line_bboxes.append((line_x0, line_y0, line_x1, line_y1))
        line_x0 = math.floor(line_x0 * width)
        line_y0 = math.floor(line_y0 * height)
        line_x1 = math.ceil(line_x1 * width)
        line_y1 = math.ceil(line_y1 * height)
        img = image[line_y0: line_y1, line_x0:line_x1]
        storeImages(section_name, image_name.split(".")[-2], "line_data", f"{line_idx}", img)
        cv2.rectangle(image, (line_x0, line_y0),
                      (line_x1, line_y1), (0, 144, 250), 30)
    # cv2.imshow("Final Registered word lines", cv2.resize(image, (720, 900)))
    # cv2.waitKey(0)
    qout.put(line_bboxes)


def data_extraction(data, image, section_name, image_name, q):
    height, width, _ = image.shape
    line_bbox = []
    all_name_x0 = [bbox[0] for bbox in data]
    all_name_y0 = [bbox[1] for bbox in data]
    all_name_x1 = [bbox[2] for bbox in data]
    all_name_y1 = [bbox[3] for bbox in data]
    name_line_bbox = [min(all_name_x0), min(all_name_y0), max(all_name_x1), max(all_name_y1)]
    line_x0, line_y0, line_x1, line_y1 = name_line_bbox
    line_bbox.append((line_x0, line_y0, line_x1, line_y1))
    line_x0 = math.floor(line_x0 * width)
    line_y0 = math.floor(line_y0 * height)
    line_x1 = math.ceil(line_x1 * width)
    line_y1 = math.ceil(line_y1 * height)
    line_width_name = 25
    name_cropped_image = image[line_y0:line_y1, line_x0:line_x1]
    for word_idx, word_data in enumerate(data):
        x, y, x1, y1 = word_data
        x = math.ceil(width * x)
        y = math.ceil(height * y)
        x1 = math.floor(width * x1)
        y1 = math.floor(height * y1)
        # remove all the vertical and horizontal lines in an img
        cv2.rectangle(image, (x, y), (x1 + 5, y1 + 5), (255, 255, 255), line_width_name)
        img = image[y-1:y1+5, x-1:x1+5]
        storeImages(section_name, image_name.split(".")[-2], "individual_data", f"{word_idx}", img)
    line_img = image[line_y0:line_y1, line_x0:line_x1]
    storeImages(section_name, image_name.split(".")[-2], "line_data", f"{0}", line_img)
    # cv2.imshow(f"Image_{section_name}", name_cropped_image)
    # cv2.waitKey(0)

    q.put(line_bbox)


