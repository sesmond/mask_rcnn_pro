#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import cv2
import os
import datetime
from mask_test import MaskTest
from m_rcnn.mask_rcnn import MaskRCNN



def cut_rectangle(box,image):
    '''
    抠出外接矩形
    :return:
    '''
    ymin = box[0][0]
    xmin = box[0][1]
    ymax = box[0][2]
    xmax = box[0][3]
    w = xmax - xmin
    h = ymax - ymin
    cut_img = image[ymin:ymin+h,xmin:xmin+w]

    return cut_img



def mask_result(test_image_file_path):
    mask_test = MaskTest()
    mask_model = MaskRCNN(train_flag=False)

    test_image_name_list = mask_test.get_images(test_image_file_path)

    for test_image_path in test_image_name_list:
        test_image_name = os.path.basename(test_image_path)

        # 读取图像
        image_info = cv2.imread(test_image_path)
        print("read img:", test_image_path)

        # Run detection
        results_info_list = mask_model.detect([image_info])
        print("results: {}".format(results_info_list))

    return results_info_list, image_info, test_image_name




def result_analysis(test_image_file_path,cut_img_path):
    results,image,test_image_name = mask_result(test_image_file_path)

    box = results[0]['rois']
    mask = results[0]['masks']

    cut_img = cut_rectangle(box, image)
    cv2.imwrite(os.path.join(cut_img_path + test_image_name),cut_img)

    return cut_img





if __name__ == "__main__":
    test_image_file_path = "data/idcard/test/input/"
    cut_img_path = "data/idcard/test/cut/"

    cut_img = result_analysis(test_image_file_path,cut_img_path)
