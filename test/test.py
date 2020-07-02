#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/18 14:42
# @Author   : WanDaoYi
# @FileName : mask_test.py
# ============================================

from datetime import datetime
import os
import colorsys
import cv2
import skimage.io
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from m_rcnn.mask_rcnn import MaskRCNN
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from config import cfg
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class MaskTest(object):

    def __init__(self):

        # 获取 类别 list
        #TODO !! 分类数据
        self.class_names_path = cfg.COMMON.COCO_CLASS_NAMES_PATH
        # self.class_names_path = cfg.COMMON.OUR_CLASS_NAMES_PATH
        self.class_names_list = self.read_class_name()

        # 测试图像的输入 和 输出 路径
        self.test_image_file_path = cfg.TEST.TEST_IMAGE_FILE_PATH
        self.output_image_path = cfg.TEST.OUTPUT_IMAGE_PATH

        # 加载网络模型
        self.mask_model = MaskRCNN(train_flag=False)
        # 加载权重模型
        self.mask_model.load_weights(cfg.TEST.COCO_MODEL_PATH, by_name=True)

        pass

    def read_class_name(self):
        with open(self.class_names_path, "r") as file:
            class_names_info = file.readlines()
            class_names_list = [class_names.strip() for class_names in class_names_info]

            return class_names_list
        pass

    def get_images(self,data_path):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        return files

    def do_test(self, show_image_flag=False):
        """
            batch predict
        :param show_image_flag: show images or not
        :return:
        """
        test_image_name_list = self.get_images(self.test_image_file_path)

        for test_image_path in test_image_name_list:
            test_image_name = os.path.basename(test_image_path)
            # test_image_path = os.path.join(self.test_image_file_path, test_image_name)
            # 读取图像
            # image_info = skimage.io.imread(test_image_path)
            image_info = cv2.imread(test_image_path)
            print("read img:",test_image_path)
            # Run detection
            results_info_list = self.mask_model.detect([image_info])
            print("results: {}".format(results_info_list))

        return results_info_list, image_info, test_image_name





if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MaskTest()
    # print(demo.class_names_list)
    demo.do_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 测试模型耗时: {}".format(end_time, end_time - start_time))

