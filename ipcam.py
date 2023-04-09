import urllib.request
import cv2
import numpy as np
import time

url = "https://philnews.ph/wp-content/uploads/2017/01/batwan-1200x1200.jpg"

def url2img(url):
    # download the image, convert it to a NumPy array, and then read into OpenCV format
    image = np.asarray(bytearray(urllib.request.urlopen(url).read()), dtype = "uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

dirc = "C://Users/Aspire ES14/Documents/11th Grade/Research/Q3/Prelim/Images/"
dirfd = 1
dirfl = 1
dirfm = "f"

#img = url2img(url)
img = cv2.imread(dirc + str(dirfd) +  "/" + str(dirfl) + dirfm + ".jpg", cv2.IMREAD_COLOR)

img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
img = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,15)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([0,128,32])
upper_range = np.array([60,255,255])

mask = cv2.inRange(img, lower_range, upper_range)
finx = cv2.bitwise_and(img, img, mask = mask)
finx = cv2.cvtColor(finx, cv2.COLOR_HSV2BGR)
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

cv2.imshow("Image", img)
cv2.moveWindow("Image", 20,20);
cv2.waitKey(0)

cv2.imshow("Mask", mask)
cv2.moveWindow("Mask", 20,20);
cv2.waitKey(0)

cv2.imshow("Final Image", finx)
cv2.moveWindow("Final Image", 20,20);
cv2.waitKey(0)


cv2.destroyAllWindows()


'''
while True:
    index = 0
    hue_avg = 0
    val_avg = 0

    #img = url2img(url)
    print(dirc + str(dirfd) +  "/" + str(dirfl) + dirfm + ".jpg")
    img = cv2.imread(dirc + str(dirfd) +  "/" + str(dirfl) + dirfm + ".jpg", cv2.IMREAD_COLOR)

    if (dirfm == "b"):
        dirfm = "f"

        dirfl += 1
        if (dirfl > 12):
            dirfd += 1
            dirfl = 1
    elif (dirfm == "f"):
        dirfm = "b"

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
'''
'''
cv2.imshow('final', cv2.resize(finx, dm, interpolation = cv2.INTER_AREA))

while True:
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
      break

cv2.destroyAllWindows()
'''

'''url = 'https://www.sample-videos.com/video/mp4/720/big_buck_bunny_720p_2mb.mp4'
cap = cv2.VideoCapture(url)

while(cap.isOpened()):
    ret, image = cap.read()    
    loadedImage = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imshow('frame',loadedImage)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
'''

'''
def transBg(img):   
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
  morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

  _, roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  mask = np.zeros(img.shape, img.dtype)

  cv2.fillPoly(mask, roi, (255,)*img.shape[2], )

  masked_image = cv2.bitwise_and(img, mask)

  return masked_image


img = url2img(url)
cv2.imshow('Image', img)
cv2.waitKey(0)
'''
