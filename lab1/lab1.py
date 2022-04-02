import numpy as np
import cv2
import imutils as im

file1 = "cover.png"

file = file1
img = cv2.imread(file)

def show(img, title="Image "):
    cv2.imshow(title+file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def compare(new_img, title="Image "):
    rez = np.hstack((img, new_img))
    cv2.imshow(title+file, rez)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

show(img)
img_gray = cv2.imread(file, 0)
cv2.imwrite("gray+"+file, img_gray)
show(img_gray, "Grayscale image ")

(h , w , d ) = img.shape
print ("width ={}, height ={}, depth ={}". format (w , h , d ) )
(B , G , R ) = img[42 , 42]
print ("R={}, G={}, B={}". format (R , G , B ) )

show(img[42:142, 42:142], "Cropped image ")
show(im.resize(img, height=420), "Resized image ")
compare(im.rotate(img, 222), "Rotated image ")
compare(cv2.GaussianBlur(img, (7,7), 0), "Blurred image ")

fig = img.copy()
cv2.rectangle(fig, (42,42), (142,142), (255, 255, 255), 5)
cv2.line(fig, (142,142), (242,242), (255, 255, 255), 3)
cv2.circle(fig, (142,142), 42, (255, 255, 255), -1)
compare(fig, "Drawing on image ")

black = np.zeros((500, 500, 3), np.uint8)
points = np.array([[100, 50], [250, 450], [400, 50], [50, 300], [450, 300]])
cv2.polylines(black, np.int32([points]), 1, (0, 0, 255), 5)
cv2.imshow("Polylines", black)
cv2.waitKey(0)
cv2.destroyAllWindows()

text = img.copy()
cv2.putText(text, "Crimson King", (10, 220), 1, 2, (0, 0, 0), 2)
compare(cv2.putText(text, "21st Century Schizoid Man",
                    (10, 240), 1, 1, (255, 255, 255), 1), "Text on image ")    