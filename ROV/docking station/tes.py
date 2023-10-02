import cv2
import numpy as np
import time


def find_button(image):
    # 将彩色图转化HSV
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 均值滤波去噪
    gray_img = cv2.blur(gray_img, (5, 5), 0)
    
    #  获取红色区域（两段H：0~30、140~180）
    lower1 = np.array([0, 46, 43])
    upper1 = np.array([25, 255, 255])
    mask1 = cv2.inRange(gray_img, lower1, upper1)
    lower2 = np.array([140, 46, 43])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(gray_img, lower2, upper2)
    mask = mask1 + mask2            #两个二值掩膜相加（0+1=1）
    # 膨胀+腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))      #8*8矩形核
    d = cv2.dilate(mask, kernel)
    new_img = cv2.erode(d, kernel)
    cv2.imshow('aa', new_img)
    # 找轮廓（参数：传入图像，检索模式: 只检测外轮廓，近似方法: 保留所有轮廓信息点）
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return -1, -1
    
    # 获取最小外接圆
    circle = new_img.copy()             #创建新图像
    for i in range(len(contours)):      #筛掉过小圆
        cnt = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(cnt)        #圆心坐标，半径
        if radius > 20:
            break
    # 画⚪
    circle = cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    cv2.imshow("circle", circle)
    return int(x), int(y)                       #圆心为img[y, x]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
    img = cv2.imread("C:/Users/13196/Desktop/a.jpg")
    x, y = find_button(img)
            
