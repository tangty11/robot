import cv2
import numpy as np
import time
import serial

focalLength = 241
# 241 cm/xiangsudian
KNOWN_length = 13
# 13.5cm

# 摄像头
ser = serial.Serial('/dev/ttyAMA0', 115200, bytesize=8, stopbits=1, parity="N", timeout=2.0)  # 使用树莓派的GPIO口连接串行口
ser.isOpen()
# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）

cap = cv2.VideoCapture(0)
width = 640  # 定义摄像头获取图像宽度
height = 480  # 定义摄像头获取图像长度

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置长度


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 320, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)  # 象征内容，栏名称，最大值，最小值，用于调用的函数
cv2.createTrackbar("Hue Max", "TrackBars", 9, 179, empty)  # 色度，饱和度，亮度
cv2.createTrackbar("Sat Min", "TrackBars", 59, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


def find_marker(image):
    # 将彩色图转化HSV
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 均值滤波去噪
    gray_img = cv2.blur(gray_img, (5, 5), 0)
    #  获取红色区域
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(gray_img, lower, upper)
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)
    cv2.imshow("mask", mask)
    # 找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 获取外接圆
    if len(contours) == 0:
        cv2.imshow("circle", image)
        return -1, -1, -1
    circle = mask.copy()
    for i in range(len(contours)):
        cnt = contours[i]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 3:
            break
        # circle = mask.copy()
        # 画⚪
        # circle = cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 3)

    # circle = mask.copy()
    # 画⚪
    circle = cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 3)
    # 最小外接⚪圆心位置

    print(int(x))
    cv2.imshow("circle", circle)

    # 第二张图的获取
    # 转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 大津法二值化
    retval, dst = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    # 膨胀,第二张图dst
    cv2.imshow("dst", dst)

    dst = cv2.dilate(dst, kernel)

    # 计算长度

    # cv2.imshow("dst", dst)
    # cv2.imshow("mask", mask)

    return int(x), int(y), dst
    # 定义距离函数


def distance_to_camera(perWidth):
    if perWidth == 0:
        return 11
    return (KNOWN_length * focalLength) / perWidth


# 计算摄像头到物体的距离


while (1):

    ret, frame = cap.read()
    # 图像中的长度
    x, y, dst = find_marker(frame)
    if (x, y, dst) == (-1, -1, -1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # x y相反

    if dst[:, x] is None:
        continue
    color = dst[:, x]
    white_count = np.sum(color == 255)
    max_count = white_count
    max_color = color
    for i in range(20):
        color = dst[:, x - 10]
        white_count = np.sum(color == 255)
        if white_count > max_count:
            max_color = color
            max_count = white_count

            # time.sleep(0.5)
    # 已知图长，实长，焦距求距离
    juli = distance_to_camera(max_count)
    # time.sleep(0.5)
    print(max_count, juli)
    if juli < 5:
        print("stop")
        ser.write(b"s")
        time.sleep(0.5)
    else:
        ser.write(b"g")
        time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 本循环内要进行调整
