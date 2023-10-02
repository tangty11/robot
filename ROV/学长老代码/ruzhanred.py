import cv2
import numpy as np
import time
#import serial
import matplotlib.pyplot as plt

focalLength = 10
KNOWN_length = 8

#ser = serial.Serial('/dev/ttyAMA0', 9600, bytesize=8, stopbits=1, parity="N", timeout=2.0)  # 使用树莓派的GPIO口连接串行口
#ser.isOpen()

# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）

cap = cv2.VideoCapture(1)


def find_marker(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    img = cv2.GaussianBlur(img, (9, 9), 0)
    # Sobel边缘检测
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 出现边缘轮廓

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    # 两个图像结合
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 膨胀腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.erode(dst, kernel)
    cv2.imshow("dst", dst)
    # 判断圆形
    # ,方法，dp,两圆最短距离，像素值大于该值会检测为边缘，越大检测的⚪更接近⚪，最小半径，最大半径
    circles1 = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=60, minRadius=50, maxRadius=20000)
    if circles1 is None:
        return -1
    circles = circles1[0, :, :]  # 提取为二维

    circles = np.uint16(np.around(circles))  # 四舍五入，取整
    dst = cv2.circle(image, (circles[0][0], circles[0][1]), circles[0][2], (255, 0, 0), 5)  # 定圆

    # (x, y), radius = cv2.minEnclosingCircle(cnt)
    # dst = img.copy()
    # 画⚪
    # dst = cv2.circle(dst, (int(x), int(y)), int(radius), (0, 0, 255), 3)
    cv2.imshow("dst", dst)
    cv2.imshow("image", image)

    # 最小外接⚪半径
    # print(radius)
    # return radius
    print(circles[0][2])
    return circles[0][2]


# 定义距离函数
def distance_to_camera(perWidth):
    return (KNOWN_length * focalLength) / perWidth


# 计算摄像头到物体的距离

while (1):
    ret, frame = cap.read()
    # 图像中的长度
    perWidth = find_marker(frame)
    if perWidth == -1:
        continue;
    # 已知图长，实长，焦距求距离
    juli = distance_to_camera(perWidth)
    print(juli)
    if juli < 10:
        ser.write(b"stop")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 本循环内要进行调整

cap0.release()
cv2.destroyAllWindows()