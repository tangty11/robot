import sys
import cv2

def imgstitcher(imgs):
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    (status, pano) = stitcher.stitch(img_l)
    if status != cv2.Stitcher_OK:
        print("不能拼接图片, error code = %d" % status)
        sys.exit(-1)
    else:
        print('111')
        cv2.imshow('pano', pano)
        
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    i = 0
    img_l = []
    while True:
        if i % 10 == 0:
            ret,frame = cap.read()
            frame = cv2.resize(frame, (400, 300))
            cv2.imshow('capture', frame)
            img_l.append(frame)
            #cv2.imwrite(str(i) + '.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    imgstitcher(img_l)
    cap.release()
