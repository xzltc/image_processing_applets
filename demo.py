# -*- coding = utf-8 -*-
# @Time :2021/6/12 8:58 下午
# @Author: XZL
# @File : demo.py
# @Software: PyCharm
import cv2 as cv
import numpy as np


def threshold_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 全局阈值二值化 主要用：cv.THRESH_OTSU和cv.THRESH_TRIANGLE两种
    # 这时p2为自动所以指定为0
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print(ret)
    cv.imshow('binary_otsu', binary)
    return binary


def canny_demo(image):
    blur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊去噪声
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # x方向梯度
    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # y方向梯度
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # dege 后面两个比值一般是3：1或者2：1
    blue_c, green_c, red_c = cv.split(blur)
    edge_canny_one = cv.Canny(green_c, 30, 150)
    # cv.canny:p1:8bit的图像，就uint8型 p2:
    cv.imshow('Canny_dege', edge_canny_one)
    return edge_canny_one


def contours_demo(image):
    """
    image_G = cv.GaussianBlur(img, (0, 0), 1)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转换灰度图
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 25, 10)
    """
    t_img = image.copy()
    dst = canny_demo(t_img)
    area = 0.0
    # p1:二值图像
    # p2：构建模型的方法 cv.RETR_TREE(树形结构)  cv.RETR_EXTERNAL(最大层轮廓)
    contours, heriachy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours 类型 list型
    for i, contour in enumerate(contours):  # 枚举 contours里面是枚举，i：index 对应contour
        # p1:原始图
        # p2:对应轮廓线
        # p3：对应的index 第几个
        # p4：轮廓线颜色
        # p5：绘制线粗细 -1:填充轮廓
        # 踩坑了注意是contours,drawContours通过index下标去找对应的contour
        cv.drawContours(t_img, contours, i, (0, 0, 255), 1)
    count_num_and_area(contours, t_img)
    # 法二 直接 p3：-1绘制全部 不然就是绘制对应的第几个轮廓
    # all_contours = cv.drawContours(image,contours,-1,(0,0,255))
    cv.imshow('contour_image', t_img)


def count_num_and_area(contours, img):
    """
    统计轮廓内的面积和标记轮廓序号
    :param contours:
    :param img:
    :return: 面积总
    """
    d_img = img.copy()
    area = 0.0
    for i, contour in enumerate(contours):
        area += cv.contourArea(contours[i])
    # 求连通域重心 以及 在重心坐标点描绘数字
    for i, j in zip(contours, range(len(contours))):
        M = cv.moments(i)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0
        draw1 = cv.putText(d_img, str(j), (cX, cY), 1, 1, (0, 255, 0), 1)  # 在中心坐标点上描绘数字
    cv.imshow('num', draw1)
    return area


def outter_contours(img):
    dst = canny_demo(img)
    t_img = img.copy()
    board = np.zeros(img.shape, np.uint8)  # 空白图像
    cnt = np.argwhere(dst >= 1)
    # 交换x:y满足opencv绘图规则
    cnt = cnt[:, [1, 0]]
    hull = cv.convexHull(cnt)
    area = cv.contourArea(hull)
    imgRet = cv.polylines(t_img, [hull], True, (0, 0, 255), 1)  # 绘制外边框多边形
    print('物件面积:' + str(area) + 'px')
    print('物件占比:' + str(round(area * 100 / (img.shape[0] * img.shape[1]), 2)) + '%')
    cv.imshow('outer', imgRet)
    return area


def large_contours(image):
    t_img = image.copy()
    dst = canny_demo(t_img)
    contours, heriachy = cv.findContours(dst, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours 类型 list型
    for i, contour in enumerate(contours):  # 枚举 contours里面是枚举，i：index 对应contour
        # p1:原始图
        # p2:对应轮廓线
        # p3：对应的index 第几个
        # p4：轮廓线颜色
        # p5：绘制线粗细 -1:填充轮廓
        # 踩坑了注意是contours,drawContours通过index下标去找对应的contour
        cv.drawContours(t_img, contours, i, (0, 0, 255), 1)
        # print(i)
        # print(contour)
    # 法二 直接 p3：-1绘制全部 不然就是绘制对应的第几个轮廓
    # all_contours = cv.drawContours(image,contours,-1,(0,0,255))
    cv.imshow('contour_image_2', t_img)


def count_white_area(img):
    binary = threshold_demo(img)
    return np.argwhere(binary == 255).shape[0]


img = cv.imread('./pic/278061623567210_.pic_hd.jpg')
h = img.shape[0]
w = img.shape[1]
img[h - 50:h, w - 50:w, :] = 0
img[h - 50:h, :50, :] = 0
# 取绿色通道
blue_c, green_c, red_c = cv.split(img)
result_img = np.concatenate((green_c, green_c, green_c), axis=-1)
cv.imshow('input', img)

t1 = cv.getTickCount()
sum_area = outter_contours(img)
white_area = count_white_area(img)
print('白色面积:' + str(white_area) + 'px')
print('占空比:' + str(round(white_area * 100 / sum_area, 2)) + '%')
# contours_demo(img)
# threshold_demo(img)
t2 = cv.getTickCount()
print('time: %s ms' % ((t2 - t1) / cv.getTickFrequency() * 1000))  # 计算运行时间
# waitkey()很重要的函数，等待键盘输入事件时间 ord()获取AsCII码
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
