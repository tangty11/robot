import cv2

def GaussFilter(path='C:/Users/86182/Desktop/pic.jpg'):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (500,300))  #resize
    img_1 = cv2.GaussianBlur(img, (5,5), 0, 1.0)
    cv2.imshow('img',img)
    cv2.imshow('img_1',img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    GaussFilter()
