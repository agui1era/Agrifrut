
import cv2  
import matplotlib.pyplot as plt 
from PIL import Image, ImageFilter 
import os
dir = '/home/oscar/Imágenes/Calibre_ML'


with os.scandir(dir) as ficheros:
    for fichero in ficheros:
        print(fichero.name)
        filename=fichero.name 
        image = Image.open(dir+"/"+filename) 
        # Converting the image to greyscale, as edge detection  
        # requires input image to be of mode = Greyscale (L) 
        image = image.convert("L") 

        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES 
        image = image.filter(ImageFilter.EDGE_ENHANCE) 

        # Saving the Image Under the name Edge_Sample.png 
        image.save("edge_"+filename)