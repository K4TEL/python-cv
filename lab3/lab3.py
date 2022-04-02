import numpy as np
import cv2
import math

file1 = "road.mp4"
    
def scale(img, scaling_factor=0.5):
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

def put_name(frame, name="Kate Lutsai"):
    return cv2.putText(frame, name, (320, 230), 1, 1, (0, 255, 255), 1)
    
def hough_test(canny_road):  
    clr_canny = cv2.cvtColor(canny_road, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLines(canny_road, 1, np.pi/180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(clr_canny, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
            
    linesP = cv2.HoughLinesP(canny_road, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(clr_canny, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
        
    return clr_canny
    
def process_image(frame):
    frame = scale(frame)
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur_filter = cv2.GaussianBlur(gray_filter, (5,5), 0)
    canny_filter = cv2.Canny(blur_filter, 50, 150)
    
    x1 = 125
    x2 = 290
    y1 = 150
    y2 = 190
    y3 = 220
    y4 = 230
    imshape = frame.shape
    road = np.array([[(0,y3), (0,y2), (x1, y1), (x2, y1), (imshape[1], y3), (imshape[1], y4)]], dtype=np.int32)
    
    mask = np.zeros_like(canny_filter)
    ignore_mask_color = 255
    cv2.fillPoly(mask, road, ignore_mask_color)
    masked_edges = cv2.bitwise_and(canny_filter, mask)
    
    rho = 1
    theta = np.pi/180
    threshold = 15
    min_line_lenght = 50
    max_line_gap = 10
    line_image = np.copy(frame)*0
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_lenght, max_line_gap)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1,y1), (x2,y2), (0,0,255), 10)
        
    return put_name(cv2.addWeighted(frame, 0.7, line_image, 1, 0))

file = file1

cv2.startWindowThread()
cap = cv2.VideoCapture(file)
frame_cntr = 0

while True:
    ret, frame = cap.read()
    if ret:
        frame_cntr += 1
        cv2.imshow("Video of road ", process_image(frame))
        if (cv2.waitKey(1) & 0XFF==ord('q')):
            break
    else:
        break
    
print(f'{frame_cntr} of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames processed')
cap.release()
cv2.destroyAllWindows()

