import numpy as np
import cv2

file0 = "faces.jpg"
file1 = "faces1.jpg"
file2 = "elon.jpg"
file3 = "people.jpg"
file4 = "people.mp4"
file5 = "face.mp4"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def show(img, title="Image "):
    cv2.imshow(title+file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def scale(file, scaling_factor=0.5):
    img = cv2.imread(file)
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
def draw_face_rects(img, face_rects):   
    for(x,y,w,h) in face_rects:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        
    show(img, "Faces on image ")
    print(f'Found {len(face_rects)} faces')
        
def draw_face_details(img): 
    gray_filter = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_filter, 7, 4)
    
    for(x,y,w,h) in faces: 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        roi_gray = gray_filter[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray)
        eye = eye_cascade.detectMultiScale(roi_gray)
        for(sx,sy,sw,sh) in smile: 
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,255,0), 1)
        for(ex,ey,ew,eh) in eye: 
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 1)
            
    show(img, "Face details on image ")
    
def draw_people_rects(img, people_rects):   
    for(x,y,w,h) in people_rects[0]:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
    show(img, "People on image ")
    print(f'Found {len(people_rects[0])} people')
    
def video_people(file):
    cv2.startWindowThread()
    cap = cv2.VideoCapture(file)
    
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        frame = cv2.resize(frame, (304, 540))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(gray_filter, winStride=(8,8))
        boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
        
        for (xa,ya,xb,yb) in boxes:
            cv2.rectangle(frame, (xa,ya), (xb,yb), (0,0,255), 2)
            
        cv2.imshow("Video of people ", frame)
        if (cv2.waitKey(1) & 0XFF==ord('q')):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
def video_face(file):
    cv2.startWindowThread()
    cap = cv2.VideoCapture(file)
    
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        frame = cv2.resize(frame, (424, 240))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes = face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5)
        boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
        
        for (xa,ya,xb,yb) in boxes:
            cv2.rectangle(frame, (xa,ya), (xb,yb), (0,0,255), 2)
            
        cv2.imshow("Video of face ", frame)
        if (cv2.waitKey(1) & 0XFF==ord('q')):
            break
        
    cap.release()
    cv2.destroyAllWindows()

file = file1
img = scale(file)
face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
draw_face_rects(img, face_rects)

file = file0
img = scale(file)
face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
draw_face_rects(img, face_rects)

file = file2
img = cv2.imread(file)
draw_face_details(img)

file = file3
img = scale(file)
people_rects = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.06)
draw_people_rects(img, people_rects)

file = file4
video_people(file)

file = file5
video_face(file)
