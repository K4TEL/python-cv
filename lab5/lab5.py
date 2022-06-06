from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os
from zipfile import ZipFile
import cv2
import numpy as np

model1 = 'resnet50_imagenet_tf.2.0.h5'
model2 = 'resnet50_coco_best_v2.1.0.h5'
model3 = 'yolo.h5'
image = "cat.jpg"
video = "video.MP4"
fruits = "archive.zip"

files = ["cherries7.jpg",
         "kachi.jpg",
         "limes.jpg",
         "tangelo.jpg",
         "salak2.jpg"]

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, model1))
prediction.loadModel()

img_detector = ObjectDetection()
img_detector.setModelTypeAsRetinaNet()
img_detector.setModelPath(os.path.join(os.getcwd(), model2))
img_detector.loadModel()

vid_detector = VideoObjectDetection()
vid_detector.setModelTypeAsYOLOv3()
vid_detector.setModelPath( os.path.join(execution_path , model3))
vid_detector.loadModel()

zip = ZipFile(fruits, "r")
for name in files:
    for file in zip.namelist():
        if name in file:
            data = zip.read(file)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            zip.extract(file)
            extracted = "fruits-360_dataset/fruits-360/test-multiple_fruits/"+name
            print(extracted)
            predictions, percentage_probabilities = prediction.classifyImage(extracted, result_count = 5)

            for i in range(len(predictions)):
                print(predictions[i], ":", percentage_probabilities[i])

detections = img_detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, image),
    output_image_path=os.path.join(execution_path, "image_detected.jpg"))

for eachObject in detections:
    print(eachObject["name"], ":", eachObject["percentage_probability"])

video_path = vid_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, video),
                                             output_file_path=os.path.join(execution_path, "video_detected"),
                                             frames_per_second=10, log_progress=True)
print(video_path)
