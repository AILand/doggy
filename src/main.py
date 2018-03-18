import tkinter as tk
from tkinter import *
import sys
import argparse
import numpy as np
from tkinter import filedialog
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from predictfd import Predict


target_size = (229, 229) #fixed size for InceptionV3 architecture


class App(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()


# create the application
#myapp = App()


def chooseimage():
    root = Tk()
    root.withdraw() 
    filename = filedialog.askopenfilename()
    img2 = Image.open(filename)
    model = load_model("models/87.2_model.hdf5")
    blah = Predict()
    preds = blah.predict(model, img2, target_size)
    blah.plotpreds(img2, preds)

    pass

#
# here are method calls to the window manager class
#
mw = tk.Tk()
#tk.Tk()
#myapp.master.title("My Do-Nothing Application")
#myapp.master.maxsize(1000, 400)
#myapp.master.g


#If you have a large number of widgets, like it looks like you will for your
#game you can specify the attributes for all widgets simply like this.
mw.option_add("*Button.Background", "lightgrey")
mw.option_add("*Button.Foreground", "blue")

mw.title('Doggy Predict')
#You can set the geometry attribute to change the root windows size
mw.geometry("500x500") #You want the size of the app to be 500x500
mw.resizable(0, 0) #Don't allow resizing in the x or y direction
mw.maxsize(1000, 400)


back = tk.Frame(master=mw,bg='white')
back.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
back.pack(fill=tk.BOTH, expand=1) #Expand the frame to fill the root window


#Changed variables so you don't have these set to None from .pack()
go = tk.Button(master=back, text='Choose Image', command=chooseimage)
go.pack()
close = tk.Button(master=back, text='Quit', command=mw.destroy)
close.pack()
info = tk.Label(master=back, text='Made by me!', bg='red', fg='black')
info.pack()

# start the program
#myapp.mainloop()
mw.mainloop()