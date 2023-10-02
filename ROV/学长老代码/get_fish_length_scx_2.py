import cv2
import numpy as np


def empty(a):
    pass


# 图像前期处理

def color_filter(img, Lower, Upper, k_e, k_d):  # k_e为腐蚀处理的卷积核，k_d为膨胀处理的卷积核
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if Lower[0] > Upper[0]:
        # 第一个参数：hsv指的是原图
        # 第二个参数：lower指的是图像中低于这个lower的值，图像值变为0
        # 第三个参数：upper指的是图像中高于这个upper的值，图像值变为0
        # 而在lower～upper之间的值变成255
        mask1 = cv2.inRange(HSV_img, Lower, 180)
        mask2 = cv2.inRange(HSV_img, 0, Upper)
        mask = cv2.bitwise_and(mask1, mask2)  # 掩膜mask1和掩膜mask2进行与操作
        # 对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
        # 利用掩膜（mask）进行“与”操作，即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已。
    else:
        mask = cv2.inRange(HSV_img, Lower, Upper)

        kernel_e = np.ones((k_e, k_e), np.uint8)  # 卷积核设置
        erosion = cv2.erode(mask, kernel_e, iterations=3)  # 腐蚀处理
        kernel_d = np.ones((k_d, k_d), np.uint8)  # 卷积核设置
        dilation = cv2.dilate(erosion, kernel_d, iterations=3)  # 膨胀处理

        img_maskcolor = dilation

    return img_maskcolor


# 根据像素值、距离、焦距测长度

def Length_C(KNOWN_DISTANCE, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    longth = (KNOWN_DISTANCE * perWidth * 2) / focalLength
    return longth


def find_marker(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), )
    cv2.imshow('img', img)
    for contour1 in contours:
        area = []
        # 找到最大的轮廓
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))

        # print(max_idx)

        # 找最小外接圆

        (x, y), radius = cv2.minEnclosingCircle(contours[max_idx])
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("show", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 最小外接圆半径

        radius = int(radius)
        print(radius)
        return radius


if __name__ == '__main__':
    distance = 0
    #cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    #设置摄像头分辨率
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #ret, frame = cap.read()
    #滑动条设置
    frame = cv2.imread('C:/Users/13196/Desktop/image.jpg' )
    cv2.namedWindow('depth')
    cv2.resizeWindow('depth', 640, 360)
    cv2.createTrackbar('Hue min', 'depth', 67, 179, empty)#65
    cv2.createTrackbar('Hue max', 'depth', 95, 179, empty)#123
    cv2.createTrackbar('Sat min', 'depth', 105, 255, empty)#32
    cv2.createTrackbar('Sat max', 'depth', 239, 255, empty)#115
    cv2.createTrackbar('Val min', 'depth', 24, 255, empty)#123#
    cv2.createTrackbar('Val max', 'depth', 255, 255, empty)#255
    cv2.createTrackbar('Kernel_e', 'depth', 1, 10, empty)#1
    cv2.createTrackbar('Kernel_d', 'depth', 1, 10, empty)#1
    while True:
        #ret, frame = cap.read()

        # 阈值及卷积核设置

        h_min = cv2.getTrackbarPos('Hue min', 'depth')  # 配合cv2.createTrackbar使用，用于获取滑块位置
        h_max = cv2.getTrackbarPos('Hue max', 'depth')
        s_min = cv2.getTrackbarPos('Sat min', 'depth')
        s_max = cv2.getTrackbarPos('Sat max', 'depth')
        v_min = cv2.getTrackbarPos('Val min', 'depth')
        v_max = cv2.getTrackbarPos('Val max', 'depth')
        Lower = np.array([h_min, s_min, v_min])
        # print(Lower)
        Upper = np.array([h_max, s_max, v_max])
        # print(Upper)
        k_e = cv2.getTrackbarPos('Kernel_e', 'depth')  # 腐蚀处理的卷积核
        k_d = cv2.getTrackbarPos('Kernel_d', 'depth')  # 膨胀处理的卷积核
        # print(k_e)
        # print(k_d)
        # 图像处理

        img_binary = color_filter(frame, Lower, Upper, k_e, k_d)
        img_binary_split = np.hsplit(img_binary, 2)
        img_binary_1_split = img_binary_split[0]
        img_binary_2_split = img_binary_split[1]
        sample_flag = 1
        if sample_flag == 1:
            img_binary_1_split_down = cv2.pyrDown(img_binary_1_split)
            img_binary_2_split_down = cv2.pyrDown(img_binary_2_split)
            # 是否进行下采样
            contours1, _ = cv2.findContours(img_binary_1_split_down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours2, _ = cv2.findContours(img_binary_2_split_down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            x_array_1 = []
            y_array_1 = []
            len_array_1 = list()
            x_array_2 = []
            y_array_2 = []
            len_array_2 = list()
            # 求形心
            for contour1 in contours1:
                if cv2.contourArea(contour1) > 50:
                    M = cv2.moments(contour1)  # 求各种矩，这些矩将用于求形心坐标
                    # print("len:" + str(len(approx)))
                    # cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)  # 绘制轮廓线
                    # if M['m00'] < 0.5:
                        #continue
                    x = int(M['m10'] / M['m00'])  # 廓线坐标
                    y = int(M['m01'] / M['m00'])  # 廓线坐标
                    if (y < 360 * (2 - sample_flag)) & (y > 120 * (2 - sample_flag)) & (x > 120 * (2 - sample_flag)) & (
                            x < 520 * (2 - sample_flag)) & (cv2.contourArea(contour1) > 100):
                        x_array_1.append(x)  # append()函数用于在列表末尾添加新的对象。
                        y_array_1.append(y)
                        cv2.circle(img_binary_1_split_down, (x, y), 10, 255)  # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]),根据给定的圆心和半径等画圆
            x_array_1_ch = np.array(x_array_1)
            for contour2 in contours2:
                if cv2.contourArea(contour2) > 50:
                    M = cv2.moments(contour2)
                    if M['m00'] < 0.5:  # 何用之有？？
                        continue
                    x = int(M['m10'] / M['m00'])  # 廓线坐标
                    y = int(M['m01'] / M['m00'])  # 廓线坐标
                    if (y < 360 * (2 - sample_flag)) & (y > 120 * (2 - sample_flag)) & (x > 120 * (2 - sample_flag)) & (
                            x < 520 * (2 - sample_flag)) & (cv2.contourArea(contour2) > 100):#1-120；3-100
                        x_array_2.append(x)
                        y_array_2.append(y)
                        cv2.circle(img_binary_2_split_down, (x, y), 10, 255)
            x_array_2_ch = np.array(x_array_2)
            print("len", len(x_array_1_ch))
            print("len", len(x_array_2_ch))
            cv2.imshow("bino1", img_binary_1_split_down)
            cv2.imshow("bino2", img_binary_2_split_down)
            if ((len(x_array_1_ch) - len(x_array_2_ch)) == 0) & (len(x_array_2_ch) != 0):
                if len(x_array_2_ch) == 1:
                    for i in x_array_1_ch - x_array_2_ch:  # i是视差，f是焦距，B是baseline基线长度distance = fB/i
                        baseline = 58.1047
                        focalLength = 490  # 焦距
                        distance = baseline*focalLength / i  # 2000为比例系数，可根据不同环境来进行调整

                else:
                    distance = - 1  # 不对齐错误

        print(distance)
        image = img_binary_1_split_down
        cv2.imshow('img', frame)
        # 储存图像及距离
        key = cv2.waitKey(200)
        if (distance != -1) & (key == ord('s')):
            cv2.imwrite('image.jpg', image)
            break
img = cv2.imread('image.jpg')
KNOWN_DISTANCE1 = distance
marker = find_marker(img)  # 轮廓像素值
lenth = Length_C(KNOWN_DISTANCE1, focalLength, marker)  # 长度
print(lenth)
