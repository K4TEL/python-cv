from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
import os

model1 = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
model2 = 'resnet50_coco_best_v2.0.1.h5'

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

for file in files:
    print(file)
    predictions, percentage_probabilities = prediction.predictImage(file, result_count = 5)
    
    for i in range(len(predictions)):
        print(predictions[i], ":", percentage_probabilities[i])
        
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(os.getcwd(), model2))

detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "cat.jpg"), 
    put_image_path=os.path.join(execution_path, "cat_new.jpg"))

for eachObject in detections:
    print(eachObject["name"], ":", eachObject["percentage_probability"])