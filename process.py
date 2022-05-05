import PIL
from PIL import Image
import cv2
import numpy as np
import glob, os


#RGB
DIM = (1280, 1024)
K_1 = np.array([[702.6030497884977, 0., 644.9296487349911], [0., 703.4541726858521, 526.4572414665469], [0., 0., 1.]])
D_1 = np.array([[0.5424763304847061, -1.7515099175022195, 6.050489512760127,-5.786170959900578]])
#Thermal 
#DIM = (640, 512)
#K_1 = np.array([[445.34173838383924, 0., 310.74708274781557], [0., 446.40695195454197, 249.54892754326676], [0., 0., 1.]])
#D_1 = np.array([[0.248994091117758, -2.909390816436668, 8.052903070857168, -7.363581411738435]])


folder="/home/youssef/Desktop/images preProcessing/results" # results directory
count=0
for infile in glob.glob("*.png"):

	pil_image = Image.open(infile).convert('RGB')
	cv_img = np.array(pil_image) 
	map1_1, map2_1 = cv2.fisheye.initUndistortRectifyMap(K_1, D_1, np.eye(3), K_1, DIM, cv2.CV_16SC2)
	undistorted_img = cv2.remap(cv_img, map1_1, map2_1, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  
	new="%06d" % count
	count=count+1
	cv2.imwrite(folder+"/"+new+".png", undistorted_img)
