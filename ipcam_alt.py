import urllib.request
import cv2
import numpy as np
import time
'''
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	image = np.asarray(bytearray(urllib.request.urlopen(url).read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

# initialize the list of image URLs to download
urls = [
	"https://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png",
	"https://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png",
	"https://www.pyimagesearch.com/wp-content/uploads/2014/12/adrian_face_detection_sidebar.png",
]
 
# loop over the image URLs
for url in urls:
	# download the image URL and display it
	print("downloading %s" % (url))
	image = url_to_image(url)
	cv2.imshow("Image", image)
	cv2.waitKey(0)


def url2img(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	image = np.asarray(bytearray(urllib.request.urlopen(url).read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

def ReadCamera(Camera):
    while True:
        cap = cv2.VideoCapture(Camera)
        (grabbed, frame) = cap.read()
        if grabbed == True:
            yield frame

print(cv2.__version__)
while True:
    img = url2img(r"http://192.168.1.4:8080/photo.jpg")
    cv2.imshow('Image', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
image = cv2.imread(r'D:\Files\Programming\Python\Research\01.png')
b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0

# RGB - Blue
cv2.imshow('B-RGB', b)
cv2.waitKey(0)

# RGB - Green
cv2.imshow('G-RGB', g)
cv2.waitKey(0)

# RGB - Red
cv2.imshow('R-RGB', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#url = r"https://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png"
url = "http://192.168.1.4:5554/back"
#url = 'https://billwurtz.com/wild-frolicking-adventures-of-informational-education.mp4'

vde = cv2.VideoCapture(url)

while 1:
    success, frame = vde.read()
    if success:
        cv2.imshow('frame',frame)
    else:
        print("Waiting for output...\r")
    cv2.waitKey(0)

vde.release()
cv2.destroyAllWindows()
'''
