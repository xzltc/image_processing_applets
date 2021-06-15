# -*- coding = utf-8 -*-
# @Time :2021/6/13 1:27 下午
# @Author: XZL
# @File : release.py
# @Software: PyCharm
import cv2 as cv
import numpy as np
import sys
import os

file_path = os.getcwd()


def threshold_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 全局阈值二值化 主要用：cv.THRESH_OTSU和cv.THRESH_TRIANGLE两种
    # 这时p2为自动所以指定为0
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print(ret)
    # cv.imshow('binary_otsu', binary)
    cv.imwrite(file_path + '/binary.jpg', binary, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    return binary


def canny(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊去噪声
    # dege 后面两个比值一般是3：1或者2：1
    blue_c, green_c, red_c = cv.split(blur)
    edge_canny_one = cv.Canny(green_c, 30, 150)
    return edge_canny_one


def outter_contours(img):
    dst = canny(img)
    t_img = img.copy()
    cnt = np.argwhere(dst >= 1)
    # 交换x:y满足opencv绘图规则
    cnt = cnt[:, [1, 0]]
    hull = cv.convexHull(cnt)
    area = cv.contourArea(hull)
    imgRet = cv.polylines(t_img, [hull], True, (0, 0, 255), 1)  # 绘制外边框多边形
    print('物件面积:' + str(area) + 'px')
    print('物件占比:' + str(round(area * 100 / (img.shape[0] * img.shape[1]), 2)) + '%')
    cv.imwrite(file_path + '/contour.jpg', imgRet, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imshow('outer', imgRet)
    return area


def count_white_area(img):
    binary = threshold_demo(img)
    return np.argwhere(binary == 255).shape[0]


def pipline(argv):
    img = cv.imread(argv)
    h = img.shape[0]
    w = img.shape[1]
    img[h - 50:h, w - 50:w, :] = 0
    img[h - 50:h, :50, :] = 0
    # img = cv.imshow(argv)
    # 取绿色通道
    # cv.imshow('input', img)

    sum_area = outter_contours(img)
    white_area = count_white_area(img)

    print('白色面积:' + str(white_area) + 'px')
    print('占空比:' + str(round(white_area * 100 / sum_area, 2)) + '%')
    #
    # if cv.waitKey(0) == ord('q'):
    #     cv.destroyAllWindows()


if __name__ == '__main__':
    # print('绝对路径:')
    path = input('绝对路径: ')
    print(path)
    # pipline(sys.argv)
    pipline(path)
