import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy


def CalCorner(img):  # 查找变换后的图像起始列
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if all(img[row, col] == [0, 0, 0]):
                return col


# 平滑缝合部分
def OptimizeSean(img, trans, dst):
    # 左图，透视变换后的图，拼接结果
    col_t = CalCorner(trans)  # 计算透视变换后图的左边界
    width = img.shape[1] - col_t  # 重叠区域宽度
    alpha = 1  # img中的像素权重
    for row in range(img.shape[0]):
        for col in range(col_t, img.shape[1]):  # 遍历重叠区所有像素
            # 如果遇到图像trans中无像素的黑点，则img权值为1
            x = (col - col_t) / width
            if all(trans[row][col] == [0, 0, 0]):
                alpha = 1
            # img中像素的权重，与当前处理点距离重叠区域左边界的距离成反比
            # else:  # 线性权值
            #     alpha = (1 - x) ** 3
            elif x <= 0.5:  # 分段权函数
                alpha = 0.5 * ((-2 * x) ** 9) + 1
            else:
                alpha = 0.5 * ((2 - 2 * x) ** 9)

            dst[row, col] = img[row, col] * alpha + trans[row, col] * (1 - alpha)


def detect(image):
    # 转化为灰度图
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建SIFT生成器
    # descriptor是一个对象，这里使用的是SIFT算法
    descriptor = cv2.SIFT_create()
    # 检测特征点及其描述子（128维向量）
    kps, features = descriptor.detectAndCompute(image, None)
    return kps, features


# 可以看下特征点(若果发现几次运行特征点数目不同，要重新读取图片)，想看特征点就在主函数里调用这个函数
def show_points(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)
    print(f"特征点数：{len(kps)}")
    img_left_points = cv2.drawKeypoints(image, kps, image)
    # cv2.imwrite('D:\\Stitch_picture\\result\\img_point.jpg', img_left_points)
    plt.figure(figsize=(9, 9))
    plt.imshow(img_left_points)
    plt.show()


def match_keypoints(kps_left, kps_right, features_left, features_right, ratio, threshold):
    """
    kpsA,kpsB,featureA,featureB: 两张图的特征点坐标及特征向量
    threshold: 阀值

    """
    # 建立暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # 使用knn检测，匹配left,right图的特征点
    raw_matches = matcher.knnMatch(features_left, features_right, 2)
    print(len(raw_matches))
    matches = []  # 存坐标，为了后面
    good = []  # 存对象，为了后面的演示
    # 筛选匹配点
    for m in raw_matches:
        # 筛选条件
        #         print(m[0].distance,m[1].distance)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append([m[0]])
            matches.append((m[0].queryIdx, m[0].trainIdx))
            """
            queryIdx：测试图像的特征点描述符的下标==>img_keft
            trainIdx：样本图像的特征点描述符下标==>img_right
            distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
            """
    # 特征点对数大于4就够用来构建变换矩阵了
    kps_left = np.float32([kp.pt for kp in kps_left])
    kps_right = np.float32([kp.pt for kp in kps_right])
    print(len(matches))
    if len(matches) > 4:
        # 获取匹配点坐标
        pts_left = np.float32([kps_left[i] for (i, _) in matches])
        pts_right = np.float32([kps_right[i] for (_, i) in matches])
        # 计算变换矩阵(采用ransac算法从pts中选择一部分点)
        H, status = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, threshold)
        return matches, H, good
    return None


def drawMatches(img_left, img_right, kps_left, kps_right, matches, H):
    # 获取图片宽度和高度
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    """对imgB进行透视变换
    由于透视变换会改变图片场景的大小，导致部分图片内容看不到
    所以对图片进行扩展:高度取最高的，宽度为两者相加"""
    image = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
    print(img_left.shape)
    print(image.shape)
    # 初始化
    image[0:h_right, 0:w_right] = img_right
    """利用以获得的单应性矩阵进行变透视换"""
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))  # (w,h
    trans = copy.deepcopy(image)  # 透视变换结果
    """将透视变换后的图片与另一张图片进行拼接"""
    image[0:h_left, 0:w_left] = img_left
    # cv2.imshow('trans', trans)
    # cv2.imwrite('D:\\Stitch_picture\\result\\trans.jpg', trans)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return image, trans


# 去除黑边
def board(pano):
    # 全景图轮廓提取
    stitched = cv2.copyMakeBorder(pano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 轮廓最小正矩形
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(cnts[0])  # 取出list中的轮廓二值图，类型为numpy.ndarray
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 对thresh进行膨胀处理，消除黑色噪点
    # thresh = cv2.dilate(thresh, (511, 511), iterations=31)
    thresh = cv2.dilate(thresh, (11, 11), iterations=3)  # 卷积核与迭代次数依据图像大小改变
    # thresh = cv2.erode(thresh, (5, 5), iterations=5)

    # 腐蚀处理，直到minRect的像素值都为0
    minRect = mask.copy()
    sub = mask.copy()
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
        # cv2.imshow('minrec', minRect)
        # cv2.imshow('sub', sub)
        # cv2.imshow('thresh', thresh)
        # cv2.waitKey(0)

    # 提取minRect轮廓并裁剪
    cnts = cv2.findContours(minRect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    (x, y, w, h) = cv2.boundingRect(cnts[0])
    stitched = stitched[y:y + h, x:x + w]
    return stitched


if __name__ == '__main__':
    img_left = cv2.imread('D:\\Stitch_picture\\stitch3.jpg', 1)
    img_right = cv2.imread('D:\\Stitch_picture\\stitch2.jpg', 1)

    #################################################################################
    # 这一部分是因为手机拍的图太大（4620*3472），处理起来很慢，所以我给缩小到原来的四分之一，如果测试图片大小不大，这部分可以不要
    size = 4  # 缩小到原来的几分之一
    height, width = img_left.shape[:2]
    img_left = cv2.resize(img_left, (width // size, height // size), interpolation=cv2.INTER_CUBIC)
    height, width = img_right.shape[:2]
    img_right = cv2.resize(img_right, (width // size, height // size), interpolation=cv2.INTER_CUBIC)
    #################################################################################

    kps_left, features_left = detect(img_left)
    kps_right, features_right = detect(img_right)
    # show_points(img_left) #  看看提取的特征点
    matches, H, good = match_keypoints(kps_left, kps_right, features_left, features_right, 0.5, 0.99)
    img = cv2.drawMatchesKnn(img_left, kps_left, img_right, kps_right, good[:30], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis, trans = drawMatches(img_left, img_right, kps_left, kps_right, matches, H)
    OptimizeSean(img_left, trans, vis)  # 平滑缝合部分
    vis = board(vis)  # 去除图像边缘黑色部分，使用之前记得去函数体里看看膨胀处理的卷积核和迭代次数是否合适，否则有一定可能出一点小问题
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.show()
