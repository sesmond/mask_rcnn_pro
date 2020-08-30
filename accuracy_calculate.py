#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import cv2
import os
import json
import numpy as np


'''
    通过计算预测的四点和标注的四点的多边形的面积的交/标注的面积>95%，计算mask-rcnn预测结果的准确率和召回率   
    包括：类别的准确率和区域的准确率
'''

def show(img, title='无标题'):
    """
    本地测试时展示图片
    :param img:
    :param name:
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/Users/yanmeima/workspace/ocr/crnn/data/data_generator/fonts/simhei.ttf')
    plt.title(title, fontsize='large', fontweight='bold', FontProperties=font)
    plt.imshow(img)
    plt.show()

# 参考python(cv2) 求倾斜矩形（多边形）交集的面积（比）
# https://blog.csdn.net/wuguangbin1230/article/details/80609477
def calculate_iou(image, original_bboxes, prediction_bboxes):
    '''
    计算任意两个倾斜多边形的面积的交并比
    '''
    original_grasp_bboxes = np.array([original_bboxes], dtype=np.int32)
    prediction_grasp_bboxes = np.array([prediction_bboxes], dtype=np.int32)

    im = np.zeros(image.shape[:2], dtype="uint8")
    im1 = np.zeros(image.shape[:2], dtype="uint8")
    original_grasp_mask = cv2.fillPoly(im, original_grasp_bboxes, 255)
    prediction_grasp_mask = cv2.fillPoly(im1, prediction_grasp_bboxes, 255)

    mask = im
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=mask)
    #masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)

    original_area = np.sum(np.float32(np.greater(original_grasp_mask, 0)))
    #or_area = np.sum(np.float32(np.greater(masked_or, 0))) # 并集面积
    and_area = np.sum(np.float32(np.greater(masked_and, 0))) # 交集面积
    IOU = and_area / original_area

    return IOU


def read_bbox(filename,bbox_path):
    bbox_json_path = os.path.join(bbox_path + filename + ".json")
    with open(bbox_json_path, "r") as f:
        json_data = json.load(f)
        bboxes = []
        for category in json_data['shapes']:
            bbox = category['points']
            bboxes.append(bbox)

    return bboxes


def calculate_acc(test_path, original_path, prediction_path):
    files = os.listdir(test_path)
    total = 0
    correct = 0
    for file in files:
        filename, ext = os.path.splitext(file)
        if ext != ".jpg":continue
        else:
            img_path = os.path.join(test_path,file)
            img = cv2.imread(img_path)
            print(img.shape)
            original_bboxes = read_bbox(filename, original_path)
            prediction_bboxes = read_bbox(filename, prediction_path)

            for original_bbox in original_bboxes:
                total +=1
                for prediction_bbox in prediction_bboxes:
                    IOU = calculate_iou(img, original_bbox, prediction_bbox)
                    if IOU >= 0.95:
                        correct +=1
                    accuracy = correct/total

        print("预测的总的区域个数:", total)
        print("预测区域和打标区域的面积iou>=0.90的区域个数:",correct)
        print("预测的正确率:",accuracy)



if __name__ == "__main__":
    test_path = "data/djz/test/input/"
    original_path = "data/djz/original/"
    prediction_path = "data/djz/prediction/"

    # 测试
    # test_path = "/Users/yanmeima/Desktop/djz_test/input/"
    # original_path = "/Users/yanmeima/Desktop/djz_test/original/"
    # prediction_path = "/Users/yanmeima/Desktop/djz_test/prediction/"

    calculate_acc(test_path, original_path, prediction_path)