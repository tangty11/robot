import cv2
print('ready')

cap0 = cv2.VideoCapture(1+0)
#cap1 = cv2.VideoCapture(3)

#cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# width = 1280  #定义摄像头获取图像宽度
# height = 960   #定义摄像头获取图像长度
# 
# cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)  #设置宽度
# cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度

'''
cap2 = cv2.VideoCapture(4)
'''
width = 1280  #定义摄像头获取图像宽度
height = 480   #定义摄像头获取图像长度

cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)  #设置宽度
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度
cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap0.set(cv2.CAP_PROP_BRIGHTNESS, 100)

print('open')
while True:
    success0, img0 = cap0.read()
    #success1, img1 = cap1.read()
    #success2, img2 = cap2.read()
    cv2.imshow('img0', img0)
    #cv2.imshow('img1', img1)
    #cv2.imshow('img2', img2)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        print('s!')
        break
cap0.release()
cv2.destroyAllWindows()