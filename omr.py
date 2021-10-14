import cv2
import subprocess
import argparse
import numpy as np
import metadata_contours
import multiprocessing as mp
import fitz
import io
from PIL import Image
import config
import sys
import os
import square_contours
import get_bbox
import check_bbox
import nms
import math
import question_contours
import registration_number_response
import qpid_response


def displayCheckBoxes(original_image, sorted_name_word_bbox, sorted_qpid_word_bbox, sorted_reg_no_bbox, sorted_reg_no_words_bbox):
    height, width, _ = original_image.shape
    for word_idx, word_data in enumerate(sorted_name_word_bbox):
        x, y, x1, y1 = word_data
        x = math.ceil(width * x)
        y = math.ceil(height * y)
        x1 = math.floor(width * x1)
        y1 = math.floor(height * y1)
        cv2.rectangle(original_image, (x, y), (x1, y1), (0, 144, 250), 27)
        cv2.imshow("Finallee Image", cv2.resize(original_image, (720, 900)))
        cv2.waitKey(0)

    for word_idx, word_data in enumerate(sorted_qpid_word_bbox):
        x, y, x1, y1 = word_data
        x = math.ceil(width * x)
        y = math.ceil(height * y)
        x1 = math.floor(width * x1)
        y1 = math.floor(height * y1)
        cv2.rectangle(original_image, (x, y), (x1, y1), (0, 144, 250), 27)
        cv2.imshow("Finallee Image", cv2.resize(original_image, (720, 900)))
        cv2.waitKey(0)

    for word_idx, word_data in enumerate(sorted_reg_no_bbox):
        x, y, x1, y1 = word_data
        x = math.ceil(width * x)
        y = math.ceil(height * y)
        x1 = math.floor(width * x1)
        y1 = math.floor(height * y1)
        cv2.rectangle(original_image, (x, y), (x1, y1), (0, 144, 250), 27)
        cv2.imshow("Finallee Image", cv2.resize(original_image, (720, 900)))
        cv2.waitKey(0)


    for word_idx, word_data in enumerate(sorted_reg_no_words_bbox):
        x, y, x1, y1 = word_data
        x = math.floor(width * x)
        y = math.floor(height * y)
        x1 = math.floor(width * x1)
        y1 = math.floor(height * y1)
        cv2.rectangle(original_image, (x, y), (x1, y1), (0, 144, 250), 27)
        cv2.imshow("Finallee Image", cv2.resize(original_image, (720, 900)))
        cv2.waitKey(0)


def getImageData(filepath):
    doc = fitz.open(filepath)
    page = doc[0]   # page = doc.loadPage(pg_idx) pg_idx = 0
    pix = page.getPixmap()
    page_image = Image.open(io.BytesIO(pix.getImageData("png")))
    open_cv_image = np.array(page_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def show_contours(cnts, image):
    for cnt in cnts:
        x, y, x1, y1 = cnt
        cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def display_image(image):
    cv2.imshow("Name Section Image", image)
    cv2.waitKey(0)


def show_name_section(boundaries, thresh, dims, image):
    thresh_x, thresh_y = thresh
    top_left_x_abs, top_left_y_abs, bot_right_x_abs, bot_right_y_abs = boundaries
    cv2.rectangle(image, (int(top_left_x_abs * dims[0]["width"]) - thresh_x, int(top_left_y_abs * dims[0]["height"]) - thresh_y), (int(
        bot_right_x_abs * dims[0]["width"]) + thresh_x, int(bot_right_y_abs * dims[0]["height"]) + thresh_y), (0, 0, 255), 5)
    # cv2.rectangle(original_image, (int(top_left_x_abs * original_dims[0]["width"]), int(top_left_y_abs * original_dims[0]["height"])), (int(bot_right_x * original_dims[0]["width"]), int(bot_right_x_abs * original_dims[0]["height"])), (0,0,255), 5)
    cv2.imshow("Original Image", cv2.resize(image, (720, 900)))
    cv2.waitKey(0)

def show_checkBoxes(bboxes, image):
    for bbox in bboxes:
        x, y, x1, y1 = bbox
        cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 30)
    cv2.imshow("Finale", cv2.resize(image, (720, 900)))
    cv2.waitKey(0)

def show_digits_bbox(image, bbox):
    height, width, _ = image.shape
    for idx in range(len(bbox)):
        for bbox_idx in range(len(bbox[idx])):
            x, y, x1, y1 = bbox[idx][bbox_idx]
            x = math.ceil(width * x)
            y = math.ceil(height * y)
            x1 = math.floor(width * x1)
            y1 = math.floor(height * y1)
            cv2.rectangle(image, (x, y), (x1, y1), (0, 144, 255), 30)
    cv2.imshow("Digits bbox", cv2.resize(image, (720,900)))
    cv2.waitKey(0)


def main(args):
    errorFiles = []
    extension = args["input"].split(".")[-1]
    image_name = args["input"].split("/")[-1]
    print("[x] The extension of the file is", extension)
    try:
        if extension == config.allowedExtensions[0]:
            open_cv_image = getImageData(args["input"])
        if extension in config.allowedExtensions[1:]:
            open_cv_image = cv2.imread(args["input"])
        else:
            print("[d] Extension not supported !")
            errorFiles.append(args["input"])
            sys.exit(0)
    except cv2.error as e:
        errorFiles.append(args["input"])
    # original_image = open_cv_image
    image = cv2.resize(open_cv_image, (720, 900), interpolation=cv2.INTER_NEAREST)
    square_template = cv2.imread(config.square_template, 0)

    print("[x] The input image size is ", image.shape)
    print("[x] The template image size is", square_template.shape)

    original_dims = []
    dims = []
    template_dims = []

    img_height, img_width, _ = image.shape
    # orig_img_height, orig_img_width, _ = original_image.shape
    ws_template, hs_template = square_template.shape[::-1]

    # original_dims.append({"height": orig_img_height, "width": orig_img_width})
    # append the image metadata
    dims.append({"height": img_height, "width": img_width})
    template_dims.append({"height": hs_template, "width": ws_template})

    # print(
    #     f"[x] The original width of the image is {orig_img_width} and the image height is {orig_img_height}")
    print(
        f"[x] The width of the image is {img_width} and the image height is ", img_height)
    print(
        f"[x] The width of the template is {ws_template} and the image height is ", hs_template)

    # use 1 for 300 dpi add auto detect functionality
    template_image = cv2.imread(config.template_filepaths[0])
    original_image = template_image
    orig_img_height, orig_img_width, _ = original_image.shape
    original_dims.append({"height": orig_img_height, "width": orig_img_width})
    template_image = cv2.resize(template_image, (720, 900))
    gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)

    qout = mp.Queue()
    process_1 = mp.Process(target=square_contours.squareContours, args=(
        (gray, square_template, template_dims, qout, )))
    process_1.start()
    square_cnts = qout.get()

    # TO VISUALIZE THE CONTOURS UNCOMMENT THE BELOW LINE
    # show_contours(square_cnts, template_image.copy())

    process_2 = mp.Process(target=get_bbox.get_bboxes,
                           args=((square_cnts, dims, qout, )))
    process_2.start()
    boundaries, thresh = qout.get()
    thresh_x, thresh_y = thresh
    top_left_x_abs, top_left_y_abs, bot_right_x_abs, bot_right_y_abs = boundaries

    top_left_x = math.floor(top_left_x_abs * dims[0]["width"])
    top_left_y = math.floor(top_left_y_abs * dims[0]["height"])
    bot_right_x = math.ceil(bot_right_x_abs * dims[0]["width"])
    bot_right_y = math.ceil(bot_right_y_abs * dims[0]["height"])

    orig_top_left_x = math.floor(top_left_x_abs * original_dims[0]["width"])
    orig_top_left_y = math.floor(top_left_y_abs * original_dims[0]["height"])
    orig_bot_right_x = math.ceil(bot_right_x_abs * original_dims[0]["width"])
    orig_bot_right_y = math.ceil(bot_right_y_abs * original_dims[0]["height"])

    name_section_image = image[top_left_y - thresh_y:bot_right_y + thresh_y +2,
                         top_left_x - thresh_x:bot_right_x + thresh_x]

    question_image = image[bot_right_y + thresh_y:]

    # FOR VISUALIZATION PURPOSES ONLY
    # show_name_section(boundaries, thresh, dims, image.copy())
    # display_image(name_section_image)
    # display_image(question_image)

    args = (original_image.copy(), (top_left_x_abs, top_left_y_abs, bot_right_x_abs, bot_right_y_abs), qout)
    process_3 = mp.Process(target=check_bbox.check_bboxes, args=args)
    process_3.start()
    bboxes = qout.get()

    # FOR VISUALIZATION PURPOSES ONLY
    # show_checkBoxes(bboxes, original_image.copy())


    args = (bboxes, original_dims, qout)
    process_4 = mp.Process(target=metadata_contours.main, args=args)
    process_4.start()
    final_data = qout.get()
    word_bboxes =[]
    final_word_bboxes =[]
    for idx, data in enumerate(final_data):
        for word_idx, word_data in enumerate(data):
            x, y, x1, y1 = word_data["bbox_loc"]
            a = math.floor(original_dims[0]["width"] * x)
            b = math.floor(original_dims[0]["height"] * y)
            c = math.floor(original_dims[0]["width"] * x1)
            d = math.floor(original_dims[0]["height"] * y1)
            # cv2.rectangle(open_cv_image, (a, b), (c, d), (0, 144, 250), 20)
            # cv2.rectangle(original_image, (x, y), (x1, y1), (0, 140, 255), 20)
            word_bboxes.append((x, y, x1, y1))
        # cv2.imshow("Final Image", cv2.resize(original_image, (720, 900)))
        # cv2.waitKey(0)
        final_word_bboxes.append(word_bboxes)
        word_bboxes = []

    # print(final_word_bboxes)
    # print("The len is ", len(final_word_bboxes))

    sorted_name_word_bbox = sorted(final_word_bboxes[0], key=lambda word_bbox: word_bbox[0])
    sorted_qpid_word_bbox = sorted(final_word_bboxes[1], key=lambda word_bbox: word_bbox[0])
    sorted_reg_no_bbox = sorted(final_word_bboxes[2], key=lambda word_bbox: word_bbox[0])
    sorted_reg_no_words_bbox = sorted(final_word_bboxes[3], key=lambda word_bbox: word_bbox[0] + word_bbox[1] * 9)

    # FOR VISUALIZATION PURPOSES UNCOMMENT THE BELOW LINE
    # displayCheckBoxes(open_cv_image.copy(), sorted_name_word_bbox, sorted_qpid_word_bbox, sorted_reg_no_bbox, sorted_reg_no_words_bbox)



    # cv2.imshow("FILLED OMR", cv2.resize(open_cv_image, (720,900)))
    # cv2.waitKey(0)


    args = (sorted_reg_no_words_bbox, qout,)
    process_5 = mp.Process(target=metadata_contours.reg_words_bbox_extraction, args=args)
    process_5.start()
    digits_bbox = qout.get()


    # FOR VISUALIZATION PURPOSES ONLY
    # show_digits_bbox(open_cv_image.copy(), digits_bbox)

    # cv2.imshow("The image is", cv2.resize(open_cv_image, (720,900)))
    # cv2.waitKey(0)
    args = (image_name, open_cv_image.copy(), digits_bbox, qout, )
    process_6 = mp.Process(target=metadata_contours.reg_words_extraction, args=args)
    process_6.start()
    reg_words_line_bboxes = qout.get()

    args = (sorted_name_word_bbox, open_cv_image.copy(), config.names['0'], image_name, qout,)
    process_7 = mp.Process(target=metadata_contours.data_extraction, args=args)
    process_7.start()

    args = (sorted_reg_no_bbox, open_cv_image.copy(), config.names['2'], image_name, qout,)
    process_8 = mp.Process(target=metadata_contours.data_extraction, args=args)
    process_8.start()

    # args = (open_cv_image, qout)
    # process_9 = mp.Process(target=question_contours.omrResponse, args=args)
    # process_9.start()
    # result_process = qout.get()


    section_response = question_contours.omrResponse(open_cv_image.copy())

    qpid = qpid_response.extractQPID(open_cv_image.copy(), config.qpid_cords)
    registration_id = registration_number_response.getData(open_cv_image.copy())
    print("[x] Registration id is ", registration_id)
    print("[x] Final QPID Code is", qpid)
    print("[x] Section A response is:", section_response)



    process_1.join()
    process_2.join()
    process_3.join()
    process_4.join()
    process_5.join()
    process_6.join()
    process_7.join()
    process_8.join()






















if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-i',
        '--input',
        help="Path to the input image",
        required=True,
        type=str
    )
    args = vars(ap.parse_args())
    main(args)
