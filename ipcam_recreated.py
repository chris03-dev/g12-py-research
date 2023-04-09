'''
This is a recreated file.
It only recreates what would've been actually used when processing images.
'''

import urllib.request
import cv2
import numpy as np
import time

dirc = "./Images/"
dirfd = 1
dirfl = 1
dirfm = "f"

while True:
    index = 0
    hue_avg = 0
    val_avg = 0
    
    print(dirc + str(dirfd) +  "/" + str(dirfl) + dirfm + ".jpg")
    img = cv2.imread(dirc + str(dirfd) +  "/" + str(dirfl) + dirfm + ".jpg", cv2.IMREAD_COLOR)

    if dirfm == "b":
        dirfm = "f"

        dirfl += 1
        if (dirfl > 12):
            dirfd += 1
            dirfl = 1
    elif dirfm == "f":
        dirfm = "b"
    
    if dirfd > 14:
        break
    
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([0,96,32])
    upper_range = np.array([60,255,255])

    mask = cv2.inRange(img, lower_range, upper_range)
    finx = cv2.bitwise_and(img, img, mask = mask)
    finx = cv2.cvtColor(finx.copy(), cv2.COLOR_BGR2HSV)

    print(img.shape[:])

    iyb = -1
    st = ""
    for iy in range(0, finx.shape[0]):                   #Y-Coordinates
        for ix in range(0, finx.shape[1]):               #X-Coordinates
            if (finx[iy, ix, 0] > 0):
                index += 1
                hue_avg += finx[iy, ix, 0]
                val_avg += finx[iy, ix, 2]
                
            #print(finx[iy, ix, 0])

    print(hue_avg / index)
    print(val_avg / index)

    img = cv2.bilateralFilter(img, 7, 50, 50)

    dm = (finx.shape[1] * 50 // 100, finx.shape[0] * 50 // 100)
