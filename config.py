import os
import numpy as np

names = {'0': 'name_bbox', '1': 'qp_section_bbox',
         '2': 'reg_number_bbox', '3': 'reg_number_words_bbox', '4': 'sign_bbox'}

outlier_bboxes = {"name_bbox":    [(0.10972222222222222, 0.07555555555555556, 0.9138888888888889, 0.1)],
                  "qp_section_bbox":   [(0.0875, 0.10666666666666667, 0.26944444444444443, 0.35333333333333333)],
                  "reg_number_bbox":   [(0.2777777777777778, 0.10777777777777778, 0.625, 0.35555555555555557)],
                  "reg_number_words_bbox": [(0.6388888888888888, 0.1, 0.9166666666666666, 0.25)],
                  "sign_bbox": [(0.5625, 0.24888888888888888, 0.8958333333333334, 0.35777777777777775)]
                  }

template_filepaths = ["./ccs_omr/omr_scanned_600.png",
                      "./ccs_omr/omr_scanned_300.png"]

allowedExtensions = ["pdf", "jpg", "png", ]

square_template = "./ccs_omr/black_square_temp/square.jpg"


template_image_paths = ['ccs_omr/full_options_temp/omr_scanned_window_w_back.png',
                        'ccs_omr/full_options_temp/omr_scanned_window_w_back_1.png',
                        'ccs_omr/full_options_temp/omr_scanned_window_p_back_2.png',
                        'ccs_omr/full_options_temp/omr_scanned_window_p_back_3.png',
                        'ccs_omr/full_options_temp/omr_scanned_window_p_back_4.png',
                        'ccs_omr/full_options_temp/omr_scanned_window_p_back_5.png'
                        ]


points = np.array([[[0.28823529, 0.13829484], [0.28823529, 0.35072712], [0.56078431, 0.35072712], [0.56078431, 0.24379812], [0.62352941, 0.24379812], [0.6254902, 0.13829484]]])

qpid_cords = (0.09313725490196079, 0.13686911890504705, 0.2549019607843137, 0.35072711719418304)