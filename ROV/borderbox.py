import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math


def draw_rectangle(Oxyz, bbox, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1,
                             linewidth=1, edgecolor=edgecolor,facecolor=facecolor,fill=fill,
                             linestyle=linestyle)           #左上角坐标，宽，高
    currentAxis.add_patch(rect)


if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    
    img = imread('frog.jpg')
    plt.imshow(img)

    bbox1 = [60, 80, 440, 400]      #左上角（y，x），右下角（y，x）
    #bbox2 = [40.93, 141.1, 226.99, 515.73]
    #bbox3 = [247.2, 131.62, 480.0, 639.32]

    currentAxis = plt.gca()
    draw_rectangle(currentAxis, bbox1, edgecolor='r')
    #draw_rectangle(currentAxis, bbox2, edgecolor='r')
    #draw_rectangle(currentAxis, bbox3, edgecolor='r')

    plt.show()
