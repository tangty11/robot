import cv2
import numpy as np


def deal(h, w, img):
    #图像处理
    kernel = np.ones((5, 5), np.uint8)    #卷积核
    erosion = cv2.erode(img, kernel)
    ed = cv2.dilate(erosion, kernel)
    #先腐蚀后膨胀，可将外围非海草图部分的亮光（值偏255）滤掉
    
    #二值化
    for i in range(h):
        for j in range(w):
            if ed[i, j] < 100:
                ed[i, j] = 0
    return ed

def bord(h, w, img):
    #去边缘（蓝背景）
    borderW = []
    borderH = []
    h_k = 3/4*h     #列阈值（一列中若有5/6为黑则去掉该列）
    w_k = 4/5*w     #行阈值
    
    for i in range(h):
        c = 0
        for j in range(w):
            if img[i, j] < 100:
                c += 1
        if c > w_k:
            borderW.append(i)
    
    for j in range(w):
        d = 0
        for i in range(h):
            if img[i, j] < 100:
                d += 1
        if d > h_k:
            borderH.append(j)
    
    for a in range(len(borderW)-1):
        if borderW[a+1] - borderW[a] > 1/2*w:
            top = borderW[a]
            bottom = borderW[a+1]
    for b in range(len(borderH)-1):
        if borderH[b+1] - borderH[b] > 1/2*h:
            left = borderH[b]
            right = borderH[b+1]
    cut = int(h/10)
    new1 = img[top+cut : bottom-cut+1, left+cut : right-cut+1]

    #去白布边缘
    h_new = new1.shape[0]
    w_new = new1.shape[1]
    h_black = []
    w_black = []
    for i in range(h_new-1):
        for j in range(w_new-2):
            if new1[i, j] < 100 and new1[i, j+1] < 100 and new1[i, j+2] <100:
                w_black.append(j)
    for j in range(w_new-1):
        for i in range(h_new-2):
            if new1[i, j] < 100 and new1[i+1, j] < 100 and new1[i+2, j] <100:
                h_black.append(i)
    Top = min(h_black)
    Bottom = max(h_black)
    Left = min(w_black)
    Right = max(w_black)
    new = new1[Top : Bottom+1, Left : Right+1]

    return new

def color_judge(i, j, new_img):
    judge = 0
    for a in range(3):
        for b in range(3):
            if new_img[i+a-1, j+b-1] < 100:
                judge += 1
    if judge > 6:
        return 0
    else:
        return 1
    
def count(new_img):
    h = new_img.shape[0]
    w = new_img.shape[1]
    a = int(h / 8)
    b = int(w / 8)
    num = 0
    for x in range(8):
        for y in range(8):
            i = int(a/2 + x*a)
            j = int(b/2 + y*b)
            if color_judge(i, j, new_img) == 0:
                new_img[i, j] = 255             ########
                num += 1
    cv2.imshow('new', new_img)          ########
    return num

if __name__ == '__main__':
    img = cv2.imread('image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #灰度图
    H = img.shape[0]
    W = img.shape[1]

    ed = deal(H, W, img)
    new_img = bord(H, W, ed)
    num = count(new_img)
    print(num)

