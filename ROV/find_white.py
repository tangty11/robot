#通过滑动条调节分别调节H、S、V的最大值和最小值，以确定待提取颜色的HSV范围（默认为白色）

import cv2
import numpy as np


def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 400, 280)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 180, empty)  # 象征内容，栏名称，默认值，最大值，用于调用的函数
cv2.createTrackbar("Hue Max", "TrackBars", 180, 180, empty)  # 色度，饱和度，亮度
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 43, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 46, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\13196\Desktop\a.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 均值滤波去噪
    gray_img = cv2.blur(gray_img, (5, 5), 0)
    i = 0
    while True:
        print(i)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(gray_img, lower, upper)
        cv2.imshow('a', mask)
        key = cv2.waitKey(500)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
