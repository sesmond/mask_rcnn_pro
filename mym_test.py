#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/18 14:42
# @Author   : WanDaoYi
# @FileName : mask_test.py
# ============================================

from datetime import datetime
import os
import cv2
import json
import numpy as np
import logging
from m_rcnn.mask_rcnn import MaskRCNN
from config import cfg
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

logger = logging.getLogger("登记证图片识别 -- 模板匹配 + local(crnn)")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

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
        self.cut_image_path = cfg.TEST.CUT_IMAGE_PATH

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

    def do_test(self):
        """
            batch predict
        :param show_image_flag: show images or not
        :return:
        """
        test_image_name_list = self.get_images(self.test_image_file_path)

        for test_image_path in test_image_name_list:
            test_image_name = os.path.basename(test_image_path)
            image_info = cv2.imread(test_image_path)
            print("read img:",test_image_path)
            # Run detection
            results_info_list = self.mask_model.detect([image_info])
            print("results: {}".format(results_info_list))

            rois = results_info_list[0]['rois']
            class_ids = results_info_list[0]['class_ids']
            masks = results_info_list[0]['masks']
            mask_cnt = masks.shape[-1]
            CLASS_NAME = ['djz-1-1','djz-1-2','djz-2']

            result = []
            for i in range(mask_cnt):
                approx = self.cut_approx_quadrang(i, masks)
                point = self.format_convert(approx)
                class_points = {
                    "labels": CLASS_NAME[class_ids[i]-1],
                    "points": point}
                result.append(class_points)
                prediction = {'shapes': result}
                print(prediction)

                image = cv2.imread(os.path.join(self.output_image_path + test_image_name))
                cv2.polylines(image, [approx], True, (0, 255, 0), 5)
            image_path = os.path.join("data/djz/debug/" + test_image_name)
            cv2.imwrite(image_path,image_info)

            prediction_path = os.path.join("data/djz/prediction/" + test_image_name[:-4] + ".json")
            # json_str = json.dumps(prediction,cls=MyEncoder)
            # with open(prediction_path, 'w') as json_file:
            #     json_file.write(json_str)
            with open(prediction_path, "w", encoding='utf-8') as g:
                json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)

        pass

    @staticmethod
    def cut_approx_quadrang(i, masks):
        '''
        mask区域，找轮廓，求近似四边形,确定近似四边形的四点坐标
        '''
        mask = masks[:, :, i]
        logger.info("mask shape:%r", mask.shape)
        binary = mask * 255
        binary = np.array(binary)
        binary = binary.reshape(len(binary), -1)
        binary = binary.astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        #logger.info("近似四边形的四点坐标:%s", approx)

        if len(approx) < 4:
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            logger.info("少于四点调整后，近似四边形的四点坐标:%s", approx)
        if len(approx) > 4:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            logger.info("多于四点调整后，近似四边形的四点坐标:%s", approx)

        cv2.polylines(binary, [approx], True, (0, 255, 0), 10)
        cv2.imwrite("data/djz/test1.jpg", binary)

        return approx
        pass

    @staticmethod
    def format_convert(approx):
        pos = approx.tolist()
        points = []
        for p in pos:
            x1 = np.float(p[0][0])
            y1 = np.float(p[0][1])
            points.append([x1, y1])
        logger.debug("转换格式后的近似四边形的四点坐标:%s", points)
        return points
        pass


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    init_logger()

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MaskTest()
    demo.do_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 测试模型耗时: {}".format(end_time, end_time - start_time))
