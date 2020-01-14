# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, sys
from PIL import Image

file_dir = '/home/jlazo/MI_BIBLIOTECA/Datasets/BUS_project/malignant/'

for infile in os.listdir(file_dir):
    print "file : " + infile
    if infile[-3:] == "tif" or infile[-3:] == "bmp" :
       # print "is tif or bmp"
       outfile = infile[:-3] + "jpeg"
       im = Image.open(file_dir+infile)
       print "new filename : " + outfile
       out = im.convert("RGB")
       out.save(outfile, "JPEG", quality=90)