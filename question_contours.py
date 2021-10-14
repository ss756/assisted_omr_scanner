import os
import cv2
import config
import numpy as np
import nms

question_metadata = {'0': 'A', "1": 'B', "2": 'C', "3": "D"}


def omrResponse(image):
    total_cnts = []
    answerResponses = [[] for j in range(4)]
    for j in range(len(config.template_image_paths)):
        template = cv2.imread(config.template_image_paths[j], 0)
        w, h = template.shape[::-1]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.4
        loc = np.where(res >= threshold)
        cnts = []
        for pt in zip(*loc[::-1]):
            if (pt[1] + h)/image.shape[1] > 0.5:
                cnts.append((pt[0], pt[1], pt[0] + w, pt[1] + h))
            if not len(cnts):
                continue
        cnts = sorted(cnts, key=lambda ctr: ctr[0])
        for idx, cnt in enumerate(cnts):
            total_cnts.append(cnt)

    total_cnts = np.array(total_cnts)
    pick = nms.non_max_suppression_fast(total_cnts, 0.4)
    print("[x] After applying non-max-suppression, %d  question bounding boxes found ..." % (len(pick)))
    assert len(pick) == 25, "The number of Question contours should be 25"
    final_cnts = []
    for idx, cnt in enumerate(pick):
        x, y, x1, y1 = cnt
        final_cnts.append((x, y, x1, y1))
        # cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 20)
        # cv2.imshow("Final Image", cv2.resize(image, (720, 900)))
        # cv2.waitKey(0)
    final_cnts = sorted(final_cnts, key=lambda ctr: ctr[1])
    # print("The number of contours detected is", len(final_cnts))
    for ques_idx in range(25):
        x, y, x1, y1 = final_cnts[ques_idx]
        if ques_idx == 24:
            detected_image = image[y:y1 - 200, x - 20:x1 + 20]
        else:
            detected_image = image[y + 20: y1 + 20, x - 20:x1 + 20]
        option_section = detected_image[:, 441:1140]
        # cv2.imshow("Section A", option_section)
        # cv2.waitKey(0)

        option_section_1 = detected_image[:, 441:1140]
        option_section_2 = detected_image[:, 1530:2190]
        option_section_3 = detected_image[:, 2610:3240]
        option_section_4 = detected_image[:, 3690:4345]

        x1 = option_section_1.shape[1]//4
        x2 = option_section_2.shape[1] // 4
        x3 = option_section_3.shape[1] // 4
        x4 = option_section_4.shape[1] // 4
        options_image = []
        whitePixels = []
        for j in range(4):
            img = option_section[:, j * x1: (j+1)*x1]
            options_image.append(img)
            number_of_white_pix = np.sum(img >= 127)
            whitePixels.append(number_of_white_pix)
        right_option = whitePixels.index(min(whitePixels))
        # print("White pixels list is", whitePixels)
        if ques_idx % 2 == 0:
            num_ans = sum(pixels < min(whitePixels) *
                          1.04 for pixels in whitePixels)
        else:
            num_ans = sum(pixels < min(whitePixels) *
                          1.02 for pixels in whitePixels)
        # print("Number of answers are", num_ans)
        if num_ans > 1 or not num_ans:
            right_option = -1
        answerResponses[0].append(right_option)

        whitePixels = []
        if ques_idx % 2 == 0:
            num_ans = sum(pixels < min(whitePixels) * 1.04 for pixels in whitePixels)
        else:
            num_ans = sum(pixels < min(whitePixels) * 1.02 for pixels in whitePixels)
        if num_ans > 1 and not num_ans:
            right_option = -1
        answerResponses[1].append(right_option)

        whitePixels = []
        for j in range(4):
            img = option_section_3[:, j * x3: (j + 1) * x3]
            number_of_white_pix = np.sum(img >= 127)
            whitePixels.append(number_of_white_pix)
        right_option = whitePixels.index(min(whitePixels))

        if ques_idx % 2 == 0:
            num_ans = sum(pixels < min(whitePixels) * 1.04 for pixels in whitePixels)
        else:
            num_ans = sum(pixels < min(whitePixels) * 1.02 for pixels in whitePixels)
        if num_ans > 1 or not num_ans:
            right_option = -1
        answerResponses[2].append(right_option)

        whitePixels = []
        for j in range(4):
            img = option_section_4[:, j * x4: (j + 1) * x4]
            number_of_white_pix = np.sum(img >= 127)
            whitePixels.append(number_of_white_pix)
        right_option = whitePixels.index(min(whitePixels))
        if ques_idx % 2 == 0:
            num_ans = sum(pixels < min(whitePixels) * 1.04 for pixels in whitePixels)
        else:
            num_ans = sum(pixels < min(whitePixels) * 1.02 for pixels in whitePixels)
        if num_ans > 1 or not num_ans:
            right_option = -1
        answerResponses[3].append(right_option)


    return answerResponses
