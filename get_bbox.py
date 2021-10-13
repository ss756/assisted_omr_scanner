import os

def get_bboxes(cnts, dims, qout):
    assert len(cnts) == 6

    img_width = dims[0]["width"]
    img_height = dims[0]["height"]

    upper_bbox1 = cnts[0]
    x1b1, y1b1, x2b1, y2b1 = upper_bbox1
    x1 = (x1b1 + x2b1) // 2
    y1 = (y1b1 + y2b1) // 2

    upper_bbox2 = cnts[1]
    x1b2, y1b2, x2b2, y2b2 = upper_bbox2
    x2 = (x1b2 + x2b2) // 2
    y2 = (y1b2 + y2b2) // 2

    lower_bbox1 = cnts[2]
    x1b3, y1b3, x2b3, y2b3 = lower_bbox1
    x3 = (x1b3 + x2b3) // 2
    y3 = (y1b3 + y2b3) // 2

    lower_bbox2 = cnts[3]
    x1b4, y1b4, x2b4, y2b4 = lower_bbox2
    x4 = (x1b4 + x2b4) // 2
    y4 = (y1b4 + y2b4) // 2

    top_left_x_abs = min([x1, x2, x3, x4]) / img_width
    top_left_y_abs = min([y1, y2, y3, y4]) / img_height
    bot_right_x_abs = max([x1, x2, x3, x4]) / img_width
    bot_right_y_abs = max([y1, y2, y3, y4]) / img_height

    thresh_y = int(33 * img_height/1152)
    thresh_x = int(33 * img_width/864)
    threshold = (thresh_x, thresh_y)
    boundaries = (top_left_x_abs, top_left_y_abs, bot_right_x_abs, bot_right_y_abs)
    qout.put((boundaries, threshold))