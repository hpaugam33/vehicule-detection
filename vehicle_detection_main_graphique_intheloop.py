#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
import imageio

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#Import interface graphique
import tkinter as tk, threading
from tkinter.messagebox import *
from tkinter.filedialog import *
from PIL import ImageTk, Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


"""
#Ecriture
label = Label(window, text="Vehicule Detection")

#bouton
boutonClose=Button(window, text="Fermer", command=window.quit)

# entrée
value = StringVar() 
value.set("texte par défaut")
entree = Entry(window, textvariable=value, width=30)

# checkbutton
checkBouton = Checkbutton(window, text="Nouveau?")

# radiobutton
value = StringVar() 
bouton1 = Radiobutton(window, text="Oui", variable=value, value=1)
bouton2 = Radiobutton(window, text="Non", variable=value, value=2)
bouton3 = Radiobutton(window, text="Peu être", variable=value, value=3)

# liste
liste = Listbox(window)
liste.insert(1, "Python")
liste.insert(2, "PHP")
liste.insert(3, "jQuery")
liste.insert(4, "CSS")
liste.insert(5, "Javascript")

# canvas
canvas = Canvas(window, width=150, height=150, background='blue')
ligne1 = canvas.create_line(75, 0, 75, 150)
ligne2 = canvas.create_line(0, 75, 150, 75)
txt = canvas.create_text(75, 60, text="Cible", font="Arial 16 italic", fill="blue")
#Possibilities of creation for a canvas
#create_arc()        :  arc de cercle
#create_bitmap()     :  bitmap
#create_image()      :  image
#create_line()       :  ligne
#create_oval()       :  ovale
#create_polygon()    :  polygone
#create_rectangle()  :  rectangle 
#create_text()       :  texte
#create_window()     :  window

#Scale
value = DoubleVar()
scale = Scale(window, variable=value)

#Frames

# frame 1
Frame1 = Frame(window, borderwidth=2, relief=GROOVE)
Frame1.pack(side=LEFT, padx=30, pady=30)
# frame 2
Frame2 = Frame(window, borderwidth=2, relief=GROOVE)
Frame2.pack(side=LEFT, padx=10, pady=10)
# frame 3 dans frame 2
Frame3 = Frame(Frame2, bg="white", borderwidth=2, relief=GROOVE)
Frame3.pack(side=RIGHT, padx=5, pady=5)
# Ajout de labels
Label(Frame1, text="Frame 1").pack(padx=10, pady=10)
Label(Frame2, text="Frame 2").pack(padx=10, pady=10)
Label(Frame3, text="Frame 3",bg="white").pack(padx=10, pady=10)

#PanedWindow
p = PanedWindow(window, orient=HORIZONTAL)
p.pack(side=BOTTOM, expand=N, fill=BOTH, pady=0, padx=0)
p.add(Label(p, text='Volet 1', background='blue', anchor=CENTER))
p.add(Label(p, text='Volet 2', background='white', anchor=CENTER) )
p.add(Label(p, text='Volet 3', background='red', anchor=CENTER) )

#SpinBox
s = Spinbox(window, from_=0, to=10)

#Message Erreur
def callback():
    if askyesno('Titre 1', 'Êtes-vous sûr de vouloir faire ça?'):
        showwarning('Titre 2', 'Tant pis...')
    else:
        showinfo('Titre 3', 'Vous avez peur!')
        showerror("Titre 4", "Aha")

Button(text='Action', command=callback).pack()

#Barre de Menus
def alert():
    showinfo("alerte", "Bravo!")

menubar = Menu(window)

menu1 = Menu(menubar, tearoff=0)
menu1.add_command(label="Créer", command=alert)
menu1.add_command(label="Editer", command=alert)
menu1.add_separator()
menu1.add_command(label="Quitter", command=window.quit)
menubar.add_cascade(label="Fichier", menu=menu1)

menu2 = Menu(menubar, tearoff=0)
menu2.add_command(label="Couper", command=alert)
menu2.add_command(label="Copier", command=alert)
menu2.add_command(label="Coller", command=alert)
menubar.add_cascade(label="Editer", menu=menu2)

menu3 = Menu(menubar, tearoff=0)
menu3.add_command(label="A propos", command=alert)
menubar.add_cascade(label="Aide", menu=menu3)

window.config(menu=menubar)

#Change le curseur quand on survole le bouton clock
Button(window, text ="clock", relief=RAISED, cursor="clock").pack()

#Recupere valeur Input
def recupere():
    showinfo("Alerte", entree.get())
    

value = StringVar() 
value.set("Valeur")
entree = Entry(window, textvariable=value, width=30)
entree.pack()
bouton = Button(window, text="Valider", command=recupere)
bouton.pack()


#Ouvre un fichier
filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
photo = PhotoImage(file=filepath)
canvas = Canvas(window, width=photo.width(), height=photo.height(), bg="yellow")
canvas.create_image(0, 0, anchor=NW, image=photo)
canvas.pack()


#Evenements
def clavier(event):
    touche = event.keysym
    print(touche)

canvas = Canvas(window, width=500, height=500)
canvas.focus_set()
canvas.bind("<Key>", clavier)
canvas.pack()


#Exemple de fonction pour bouger un carre avec fleches
# fonction appellée lorsque l'utilisateur presse une touche
def clavier(event):
    global coords

    touche = event.keysym

    if touche == "Up":
        coords = (coords[0], coords[1] - 10)
    elif touche == "Down":
        coords = (coords[0], coords[1] + 10)
    elif touche == "Right":
        coords = (coords[0] + 10, coords[1])
    elif touche == "Left":
        coords = (coords[0] -10, coords[1])
    # changement de coordonnées pour le rectangle
    canvas.coords(rectangle, coords[0], coords[1], coords[0]+25, coords[1]+25)

# création du canvas
canvas = Canvas(window, width=250, height=250, bg="ivory")
# coordonnées initiales
coords = (0, 0)
# création du rectangle
rectangle = canvas.create_rectangle(0,0,25,25,fill="violet")
# ajout du bond sur les touches du clavier
canvas.focus_set()
canvas.bind("<Key>", clavier)
# création du canvas
canvas.pack()


liste.pack()


entree.pack()
label.pack()
checkBouton.pack()

s.pack()

bouton1.pack()
bouton2.pack()
bouton3.pack()

canvas.pack()

scale.pack()

p.pack()

boutonClose.pack()

#Relief
b1 = Button(window, text ="FLAT", relief=FLAT).pack()
b2 = Button(window, text ="RAISED", relief=RAISED).pack()
b3 = Button(window, text ="SUNKEN", relief=SUNKEN).pack()
b4 = Button(window, text ="GROOVE", relief=GROOVE).pack()
b5 = Button(window, text ="RIDGE", relief=RIDGE).pack()
"""






#Use cam or video recorded ?









#Adjust the line to the image



#Select your video from the database
VIDEO_FOLDER = 'video dataset'

#filepath = \
#    askopenfilename(title="Open your video",filetypes=[('mp4 files','.mp4'), ('mkv files','.mkv'),('avi files','.avi'),('all files','.*')])

#cap = cv2.VideoCapture(filepath)
cap = cv2.VideoCapture(0)

#frame dimension
width = cap.get(3)
height = cap.get(4)

#Frame number
frame_number = cap.get(7)

#Time of the detected object
time=  cap.get(0)






# Capture from camera
#cap = cv2.VideoCapture(0)










"""
#Scales
value = DoubleVar()
horizontalScale = Scale(configFrame, orient='horizontal', from_=0, to=10, resolution=0.1, tickinterval=2, length=300, label='Horizontal Size')
verticalScale = Scale(configFrame, orient='vertical', from_=0, to=10, resolution=0.1, tickinterval=2, length=300, label='Vertical Size')

horizontalScale.pack()
verticalScale.pack()


"""



# function for video streaming
def video_stream(frame, lmain,canvas):
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.config(image=imgtk)
    lmain.after(1, video_stream) 
    image = canvas.create_image(width/2, height/2, image=imgtk)
    line = canvas.create_line(0, height/2, width,height/2 , fill="red", width=2)

    
    






#Creation of an excel file recording detection datas
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    #Categories recorded
    csv_line = \
        'Vehicle Type/Size, Vehicle Movement Direction, Vehicle Speed (km/h), Time (s)'
    writer.writerows([csv_line.split(',')])

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )





# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'   #'ssdlite_mobilenet_v2_coco_2018_05_097'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt') 

#oid_bbox_trainable_label_map.pbtxt
#pascal_label_map.pbtxt

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            1)).astype(np.uint8)


# Detection
def object_detection_function():






    #Window creation
    window = Tk()
    window.title("Vehicule detection")
    window['bg']='white'

    # Create a frame for the video
    videoFrame= Frame(window, bg="white")
    videoFrame.pack(side=LEFT, padx=2, pady=2)
    #videoFrame.grid()
    # Create a frame for the user parameters
    configFrame = Frame(window, bg="white", borderwidth=2, relief=GROOVE)
    configFrame.pack(side=RIGHT, padx=0, pady=0)
    # Add Labels for the frames
    #Label(videoFrame, text="Vidéo Output").pack(padx=10, pady=10)
    Label(configFrame, text="User Parameters").pack(padx=10, pady=10)

    # Create a label in the frame
    lmain = Label(videoFrame)
    #lmain.grid()

    canvas = Canvas(window, width=width, height = height)
    canvas.pack()


    #Barre de Menus
    def alert():
        showinfo("alerte", "Bravo!")

    menubar = Menu(window)

    menu1 = Menu(menubar, tearoff=0)
    menu1.add_command(label="Créer", command=alert)
    menu1.add_command(label="Editer", command=alert)
    menu1.add_separator()
    menu1.add_command(label="Quitter", command=window.quit)
    menubar.add_cascade(label="Fichier", menu=menu1)

    menu2 = Menu(menubar, tearoff=0)
    menu2.add_command(label="Couper", command=alert)
    menu2.add_command(label="Copier", command=alert)
    menu2.add_command(label="Coller", command=alert)
    menubar.add_cascade(label="Editer", menu=menu2)

    menu3 = Menu(menubar, tearoff=0)
    menu3.add_command(label="A propos", command=alert)
    menubar.add_cascade(label="Aide", menu=menu3)

    window.config(menu=menubar)



    """ Organize the config frame """
    changex0 = False


    def retour():
        changex0 = True
        print(changex0)


    # Input
    lineParameters = Frame(configFrame, bg="white", borderwidth=2, relief=GROOVE)
    lineParameters.pack()

    param1 = Label(lineParameters, text="x0", bg="white")
    param1.pack( side = LEFT)


    valueX0Line = StringVar()
    x0value = Entry(lineParameters, textvariable=valueX0Line, width=20)
    x0value.pack(side = RIGHT)
    Button (lineParameters, text = "Valider", command=retour).pack(side = RIGHT)
   



    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    time_s = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                time_ms = cap.get(0)
                time_s = int(time_ms/1000)
                time = str(time_s)
                

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                #print([category_index.get(i) for i in classes[0]])
                #print(scores)

                # Visualization of the results of a detection.
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    time_s,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    )

                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                line_width = int(width);
                line_height = int(height/1.2);
                if counter == 1:
                    cv2.line(input_frame, (0, line_height), (line_width, line_height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, line_height), (line_width, line_height), (0, 0, 0xFF), 5)
                # insert information text to video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    ((int)(line_width*0.8), line_height -30),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,

                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Movement Direction: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Speed(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Vehicle Size/Type: ' + size,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Time: ' + time + ' s',
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                cv2.imshow('vehicle detection', input_frame)
                #video_stream(input_frame, lmain, canvas)

                """cv2image = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.config(image=imgtk)
                lmain.after(1, video_stream) 
                image = canvas.create_image(width/2, height/2, image=imgtk)
                line = canvas.create_line(0, height/2, width,height/2 , fill="red", width=2)"""




                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, direction, speed, time) = csv_line.split(',')

                        writer.writerows([csv_line.split(',')])

                
            cap.release()
            cv2.destroyAllWindows()
            window.mainloop()


object_detection_function()	
	
