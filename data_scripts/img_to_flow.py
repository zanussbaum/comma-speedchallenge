import cv2
import pickle
import numpy as np
import sys


table = np.empty((1,256), np.uint8)
for i in range(256):
    table[0,i] = np.clip(pow(i / 255.0, .4) * 255.0, 0, 255)
    
def brighten(img):
    res = cv2.LUT(img, table)
    return res

def canny_edge_detector(image): 
    # Convert the image color to grayscale 
    image = brighten(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
      
    # Reduce noise from the image 
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)  
    canny = cv2.Canny(blur, 50, 150) 
    return canny


def img_to_flow_train():
    cap = cv2.VideoCapture("../video/train.mp4")
    ret, frame1 = cap.read()
    prvs = canny_edge_detector(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    with open("../video/train.txt", 'r') as f:
        speeds = f.read().split('\n')

    speeds = [float(s) for s in speeds]

    num_frames = 0

    while(1):
        ret, frame2 = cap.read()
        next = canny_edge_detector(frame2)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        cv2.imwrite(f"../frames/flow/flow_{num_frames}.jpg", rgb)
        cv2.imwrite(f"../frames/vert/vert_{num_frames}.jpg", vert)
        
        num_frames += 1
        if (num_frames+1) % 100 == 0:
            print(f"Finished {num_frames} frames")
        prvs = next
        

    cap.release()
    cv2.destroyAllWindows()


def img_to_flow_test():
    cap = cv2.VideoCapture("../video/test.mp4")
    ret, frame1 = cap.read()
    prvs = canny_edge_detector(frame1)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    num_frames = 0
    while(1):
        ret, frame2 = cap.read()
        next = canny_edge_detector(frame2)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        

        cv2.imwrite(f"../frames/test_flow/flow_{num_frames}.jpg", rgb)
        
        num_frames += 1
        if (num_frames+1) % 100 == 0:
            print(f"Finished {num_frames} frames")
        prvs = next
        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        img_to_flow_train()

    else:
        if sys.argv[1] == 'train':
            img_to_flow_train()
        elif sys.argv[1] == 'test':
            img_to_flow_test()
        else:
            raise NotImplementedError("Argument must either be test or train")
