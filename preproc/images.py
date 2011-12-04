'''
Module for importing images and transforming them into array representations
to be fed to hierarchies.

Arrays are 2d and contain monochrome values for pixels between 0 and 1.
Incidentally, the paper implies that one could obtain color vision by
vectorizing the inputs to the vision hierarchy by color.

'''

import numpy as np
from PIL import Image, ImageDraw

#Our representation 
class Data_Image(object):
    
    #Import an image from a file
    def __init__(self, fh=None):
        if fh == None:
            self.im = Image.new("L", (7,7), 256)
        else:
            self.im = Image.open(fh)
    
    def data(self):
        if not hasattr(self, '_data'):
            self._data = (np.array(self.im.convert("L"))/255.0).reshape(1,49)[0]
        return self._data

#This creates a shifted square image that matches the supplied dimensions 
class Square_Image(Data_Image):
    #Generates a 7x7 white background with a black square that
    #is of width "edge_length" and has its upper left hand corner at
    #upper_left_corner
    def __init__(self, edge_length, upper_left_corner):
        self.l, self.u = edge_length, upper_left_corner
        self.im = Image.new("L", (7,7), 256)
        draw = ImageDraw.Draw(self.im)
        endpoint = (upper_left_corner[0] + edge_length-1, upper_left_corner[1] + edge_length-1)
        draw.rectangle([upper_left_corner, endpoint], fill=255, outline=0)
    
    def save(self):
        self.im.save("square_{}_{}.png".format(self.l, self.u), "PNG")

class Diamond_Image(Data_Image):
    def __init__(self, half_length, center):
        self.hl, self.c = half_length, center
        self.im = Image.new("L", (7,7), 256)
        draw = ImageDraw.Draw(self.im)
        l = (center[0]-half_length, center[1])
        ct = (center[0], center[1]-half_length)
        cb = (center[0], center[1]+half_length)
        r = (center[0]+half_length, center[1])
        draw.line([l,ct,r,cb,l], fill=0)


    def save(self):
        self.im.save("diamond_{}_{}.png".format(self.hl, self.c), "PNG")

class X_Image(Data_Image):
    def __init__(self, edge_length, upper_left_corner):
        self.l, self.u = edge_length, upper_left_corner
        self.im = Image.new("L", (7,7), 256)
        draw = ImageDraw.Draw(self.im)
        lt = self.u
        lb = (self.u[0], self.u[1]+edge_length-1)
        rt = (self.u[0]+edge_length-1, self.u[1])
        rb = (self.u[0]+edge_length-1, self.u[1]+edge_length-1)
        draw.line([lt,rb], fill=0)
        draw.line([rt,lb], fill=0)

    def save(self):
        self.im.save("x_{}_{}.png".format(self.l, self.u), "PNG")

