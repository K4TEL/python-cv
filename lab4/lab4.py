import dlib
from skimage import io
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup

sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()

file1 = "id.jpg"
file2 = "portrait.jpg"

lampert_page = "https://ist.ac.at/en/research/lampert-group/"
lampert_photo = "https://scontent.fiev8-2.fna.fbcdn.net/v/t1.18169-9/10423908_867952099923880_3146502396159739905_n.jpg?_nc_cat=110&ccb=1-5&_nc_sid=973b4a&_nc_ohc=CzcKy_GwKUAAX990Ur4&_nc_ht=scontent.fiev8-2.fna&oh=00_AT-synax7Eq9DXBomF7z7Nuoaifb1c32sTYfQxXu-Xhnpw&oe=62497DB0"

def get_descriptor(file):
    img = io.imread(file)

    win1 = dlib.image_window()
    win1.clear_overlay()
    win1.set_image(img)
    
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        print ("File: {} \nDetection {}: Left: {} Top: {} Right: {} Bottom: {}". format (
            file, k , d.left(), d.top(), d.right(), d.bottom()))
    
        shape = sp(img, d)
        win1.clear_overlay()
        win1.add_overlay(d)
        win1.add_overlay(shape)
        win1.wait_for_keypress('q')
        
        return facerec.compute_face_descriptor(img, shape) 

def detect_similar(file, people):
    img = io.imread(file)

    win1 = dlib.image_window()
    win1.clear_overlay()
    win1.set_image(img)
    
    dets = detector(img, 1)
    print(f'Знайдено {len(dets)} облич на груповому фото')
    
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor1 = facerec.compute_face_descriptor(img, shape)
        found = False
        a_min = 1
        
        for person, descriptor in people.items():
            face_descriptor2 = people[person]
            a = distance.euclidean(face_descriptor1, face_descriptor2)
        
            if a < 0.6:
                print ("Detection {}: Left: {} Top: {} Right: {} Bottom: {}". format (
                    k , d.left(), d.top(), d.right(), d.bottom()))
                print(f"Розпізнано обличчя {person}, відстань: {a}")
                
                win1.clear_overlay()
                win1.add_overlay(d)
                win1.add_overlay(shape)
                win1.wait_for_keypress('q')
                
                found = True
            else:
                if a < a_min:
                    a_min = a
        
        if not found:
            print(f"Detection {k} has no matches, min distance: {a_min}")
                
face_descriptor1 = get_descriptor(file1)
face_descriptor2 = get_descriptor(file2)
print("Відстань:", distance.euclidean(face_descriptor1, face_descriptor2))

page_data = requests.get(lampert_page).text
img_urls = [image.attrs.get("src") for image in BeautifulSoup(page_data, "lxml").find_all("img")]
img_urls = [url for url in img_urls if url and len(url) > 0]
img_urls = [url for url in img_urls if "profile" in url]

name_texts = [p.text for p in BeautifulSoup(page_data, "lxml").find_all("p", {"class": "pname"})]

Dictionary_Lampert_Group = {}
for i in range(len(name_texts)):
    person = name_texts[i]
    print(person)
    descriptor = get_descriptor(img_urls[i])
    if not descriptor:
        print("Неможливо розпізнати обличчя")
    else:
        Dictionary_Lampert_Group[person] = descriptor

detect_similar(lampert_photo, Dictionary_Lampert_Group)
