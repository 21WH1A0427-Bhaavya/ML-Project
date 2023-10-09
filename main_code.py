import pickle
import os.path

import tkinter.messagebox
from tkinter import simpledialog
from tkinter import *

import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class DrawingClassifier:

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None #count instances of the sample drawings
        self.clf - None #model
        self.proj_name = None #directory
        self.root = None #Tkinter window
        self.image1 = None #canvas into image to feed into the ML model

        self.status_label = None #which model is being used
        self.canvas = None
        self.draw = None 
        self.brush_width = 15 #default brush size

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self): #text boxes
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below", parent = msg)
        if os.path.exists(self.proj_name): #if file directory already exists
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f: #rb = read binary
                data = pickle.load(f)
            
            #extracting all information - in this pickle file we have dictionary which has key value pairs.
            #keys - c1, c2, c3 and they lead to class1, class2, class3
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']

            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']

            self.clf = data['clf']
            self.proj_name = data['pname']
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

            self.class1_counter = 1
            self.class1_counter = 1
            self.class1_counter = 1

            self.clf = LinearSVC() #default

            os.mkdir(self.proj_name) #make directory
            os.chdir(self.proj_name) #change directory
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..") #go back


    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255,255,255)

        self.root = Tk()
        self.root.title(f"Drawing Classifier - {self.proj_name}")

        #Canvas creating
        self.canvas = Canvas(self.root, width = WIDTH-10, height=HEIGHT-10, bg = "white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btn_frame = tkinter.Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

    def train_model(self):
        img_list = np.array([]) #pixels
        class_list = np.array([]) #labels

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.proj_name}/{self.class3}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter-1 + self.class2_counter-1 + self.class3_counter-1, 2500)
        self.clf.fit(img_list, class_list)
        tkinter.messagebox.showinfo("Drawing Classifier", "Model Succesfullt trained!", parent = self.root)

    def predict(self):
        #file from canvas -> store in a temp file -> reshape it to fit model shape -> predict

        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50,50), PIL.Image.ANTIALIAS)
        img.save("predictshape.png")
        img = cv.imread("predictshape.png")
        img = img.reshape(2500)
        prediction =self.clf.predict([img])
        
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class1}", parent = self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class2}", parent = self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class3}", parent = self.root)

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()

        self.status_label.config(text = f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        pass
    def load_model(self):
        pass
    def save_everything(self):
        pass
        