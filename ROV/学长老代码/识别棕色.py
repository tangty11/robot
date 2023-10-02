
import time
import cv2
import numpy as np
import serial


def empty(a):
    pass


# 摄像头
ser = serial.Serial('/dev/ttyAMA0', 115200, bytesize=8, stopbits=1, parity="N", timeout=2.0)  # 使用树莓派的GPIO口连接串行口
ser.isOpen()
# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）
cap = cv2.VideoCapture(1)
# 创建栏
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 300, 140)
cv2.createTrackbar("Hue Min", "TrackBars", 11, 179, empty)  # 象征内容，栏名称，最大值，最小值，用于调用的函数
cv2.createTrackbar("Hue Max", "TrackBars", 24, 179, empty)  # 色度，饱和度，亮度
cv2.createTrackbar("Sat Min", "TrackBars", 53, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
while (1):
    ret, frame = cap.read()
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 高斯滤波
    # imgHSV=cv2.GaussianBlur(imgHSV,(9,9),0)
    # 腐蚀
    # kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # imgHSV=cv2.erode(imgHSV,kernel)
    # 膨胀
    # imgHSV=cv2.dilate(imgHSV,kernel)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # 筛选颜色
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    # 找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    area = 0
    num = 0
    # 判断轮廓内大小
    for i in contours:
        area = cv2.contourArea(i)
        if (area > 200):
            # 轮廓
            cv2.drawContours(frame, contours, num, (255, 0, 0), 5)
            break
        num = num + 1

    # imgResult = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("original", frame)
    #   cv2.imshow("HSV",imgHSV)
    cv2.imshow("mask", mask)
    #   cv2.imshow("result",imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break