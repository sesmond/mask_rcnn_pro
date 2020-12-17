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
import base64
import numpy as np
import logging
from m_rcnn.mask_rcnn import MaskRCNN
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

logger = logging.getLogger("登记证图片识别 -- maskrcnn预测")

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
        self.debug_image_path = cfg.TEST.DEBUG_IMAGE_PATH
        self.prediction_path = cfg.TEST.PREDICTION_PATH

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
            h, w, _ = image_info.shape

            # Run detection
            results_info_list = self.mask_model.detect([image_info])
            print("results: {}".format(results_info_list))

            rois = results_info_list[0]['rois']
            class_ids = results_info_list[0]['class_ids']
            masks = results_info_list[0]['masks']
            mask_cnt = masks.shape[-1]
            CLASS_NAME = self.class_names_list[1:]

            result = []
            for i in range(mask_cnt):
                approx, hull, cut_img = self.cut_approx_quadrang(i, masks, rois, image_info)
                point = self.format_convert(hull)

                class_points = {
                    "label": CLASS_NAME[class_ids[i] - 1],
                    "points": point,
                    "group_id": " ",
                    "shape_type": "polygon",
                    "flags": {}
                }
                result.append(class_points)
                prediction = {"version": "3.16.7",
                              "flags": {},
                              'shapes': result,
                              "imagePath": test_image_name,
                              "imageData": self.nparray2base64(image_info),
                              "imageHeight": h,
                              "imageWidth": w
                              }

                rect = cv2.minAreaRect(np.array(hull))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
                box = np.int0(box)

                # 画线
                cv2.polylines(image_info, [box], True, (0, 255, 255), 3)  # 面积最小的外接矩形框box
                #cv2.polylines(image_info, [approx], True, (255, 255, 0), 3)  # 近似四边形
                #cv2.polylines(image_info, [hull], True, (0, 255, 0), 3)  # 凸包
            image_path = os.path.join(self.debug_image_path + test_image_name)
            cv2.imwrite(image_path, image_info)

            prediction_json_path = os.path.join(self.prediction_path + test_image_name[:-4] + ".json")
            with open(prediction_json_path, "w", encoding='utf-8') as g:
                json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)
        pass

    @staticmethod
    def nparray2base64(img_data):
        """
            nparray格式的图片转为base64（cv2直接读出来的就是）
        :param img_data:
        :return:
        """
        _, d = cv2.imencode('.jpg', img_data)
        return str(base64.b64encode(d), 'utf-8')

    @staticmethod
    def cut_approx_quadrang(i, masks, rois, image_info):
        '''
        1、抠出外接矩形
        2、mask区域，找轮廓，求近似四边形,确定近似四边形的四点坐标
        '''
        ymin = rois[i][0]
        xmin = rois[i][1]
        ymax = rois[i][2]
        xmax = rois[i][3]
        w = xmax - xmin
        h = ymax - ymin
        cut_img = image_info[ymin:ymin+h,xmin:xmin+w]

        mask = masks[:, :, i]
        logger.info("mask shape:%r", mask.shape)
        binary = mask * 255
        binary = np.array(binary)
        binary = binary.reshape(len(binary), -1)
        binary = binary.astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        # todo ：尝试下凸包
        # 寻找凸包并绘制凸包（轮廓）
        hull = cv2.convexHull(cnt)
        # length = len(hull)
        # for i in range(len(hull)):
        #     cv2.line(binary, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 10)
        # cv2.imwrite("data/djz/test1.jpg", binary)

        # todo: 这边绿本第一个小框会出现一条直线，还需要调整
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

        #cv2.polylines(binary, [approx], True, (0, 255, 0), 10)

        return approx, hull, cut_img
        pass


    # def cut_rectangle(self):
    #     '''
    #     抠出外接矩形
    #     :return:
    #     '''
    #     results_info_list, image_info, test_image_name = self.do_test(self.test_image_file_path)
    #
    #     box = results_info_list[0]['rois']
    #     mask = results_info_list[0]['masks']
    #     ymin = box[0][0]
    #     xmin = box[0][1]
    #     ymax = box[0][2]
    #     xmax = box[0][3]
    #     w = xmax - xmin
    #     h = ymax - ymin
    #     cut_img = image_info[ymin:ymin+h,xmin:xmin+w]
    #     cut_img_path = "data/idcard/test/cut/"
    #     cv2.imwrite(os.path.join(cut_img_path + test_image_name), cut_img)
    #
    #     pass
    #

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
