import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_subpix,corner_harris, corner_peaks,BRIEF,match_descriptors,plot_matches
import matplotlib.image as img
import cv2
import random
import math


########## read image and covert to gray ###########
image1 = img.imread("./2.jpeg")
image2 = img.imread("./1.jpeg")
image1gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

########## basic varibles and tools ################
data1x,data1y = [],[]
reversed_match_num = 100
brief_patch_size = 49
peak_patch_size = 1
iters = 10000
allowed_sigma = 0.25

########## Harris Detector #########################
# dst1 = cv2.cornerHarris(gray,2,3,0.04)
# dst2 = cv2.cornerHarris(gray,2,3,0.04)
keypoints1 = corner_peaks(corner_harris(image1gray,sigma=1),min_distance=peak_patch_size)
keypoints2 = corner_peaks(corner_harris(image2gray,sigma=1),min_distance=peak_patch_size)
print(keypoints1.size)
kp1 = [cv2.KeyPoint(keypoint[1], keypoint[0],1) for keypoint in keypoints1]
kp2 = [cv2.KeyPoint(keypoint[1], keypoint[0],1) for keypoint in keypoints2]
# cv2.drawKeypoints(image1,kp1,image1)
# plt.plot(keypoints1[:,1],keypoints1[:,0],'+r',markersize=10)

########## Brief Descriptor ########################
# extractor = BRIEF(patch_size=brief_patch_size)
# extractor.extract(image1gray,keypoints1)
# keypoints1 = keypoints1[extractor.mask]
# descriptors1 = extractor.descriptors
# plt.plot(keypoints1[:,1],keypoints1[:,0],'+r',markersize=10)

########## SIFT Detector and Descriptor ############
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1= sift.compute(image1gray,kp1)
kp2,des2= sift.compute(image2gray,kp2)
# kp1,des1 = sift.detectAndCompute(image1gray,None)
# kp2,des2 = sift.detectAndCompute(image2gray,None)
# cv2.drawKeypoints(image1,kp1,image1)

##########  BF matcher ##########################
bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
m12 = bf.match(des1,des2)
m12 = sorted(m12, key = lambda x:x.distance)
m12_reserved = m12[:reversed_match_num]
# img12_match = cv2.drawMatches(image1,kp1,image2,kp2,m12_reserved,None,flags=2)

########## skicit matcher ##########################
# matches12 = match_descriptors(descriptors1, descriptors2, metric='euclidean',cross_check=True)
#plot_matches(plt,image1, image1, keypoints1, keypoints2, matches12)

########## RANSAC ###############################
def compute_fundamental(x1, x2):
    n = x1.shape[0]
    xx,xy,yy,x,y,i,ux,uy,u,vx,vy,v = 0,0,0,0,0,0,0,0,0,0,0,0
    A = np.zeros((6, 6))
    for i in range(n):
        xx += x1[i,0]*x1[i,0]
        xy += x1[i,0]*x1[i,1]
        x += x1[i,0]
        y += x1[i,1]
        yy += x1[i,1]*x1[i,1]
        i += x1[i,2]
    A = [[xx,xy,x,0,0,0],
            [xy,yy,y,0,0,0],
            [x,y,i,0,0,0],
            [0,0,0,xx,xy,x],
            [0,0,0,xy,yy,y],
            [0,0,0,x,y,i]]
    for i in range(n):
        ux += x1[i,0]*x2[i,0]
        uy += x1[i,1]*x2[i,0]
        u += x2[i,0]
        vx +=  x1[i,0]*x2[i,1]
        vy +=  x1[i,1]*x2[i,1]
        v += x2[i,1]
    b = [[ux],[uy],[u],[vx],[vy],[v]]
    X, res, rnk, s =  np.linalg.lstsq(A, b)
    X = [[X[0,0],X[1,0],X[2,0]],
            [X[3,0],X[4,0],X[5,0]],
            [0,0,1]]
    # print("compute_fundamental")
    # print(X)
    # print("end compute_fundamental")
    return np.array(X, dtype=float)

def randSeed(set, num = 4):
    points_set = random.sample(set, num)
    return points_set

def PointCoordinates(points_set, keypoints1, keypoints2):
    x1 = []
    x2 = []
    tuple_dim = (1.,)
    for i in points_set:
        tuple_x1 = keypoints1[i.queryIdx].pt + tuple_dim
        tuple_x2 = keypoints2[i.trainIdx].pt + tuple_dim
        x1.append(tuple_x1)
        x2.append(tuple_x2)
    return np.array(x1, dtype=float), np.array(x2, dtype=float)

def inlier(X,points_set, keypoints1,keypoints2,confidence):
    num = 0
    ransac_good = []
    x1, x2 = PointCoordinates(points_set, keypoints1, keypoints2)
    X = compute_fundamental(x1,x2)
    for i in range(len(x2)):
        err = np.sqrt((x2[i].reshape(-1,1)-X.dot(x1[i].reshape(-1,1)))**2)
        # print("inlier")
        # print(err)
        # print("end inlier")
        if np.max(err) < confidence:
            ransac_good.append(points_set[i])
            num += 1
    return num, ransac_good

def ransac(matches, keypoints1, keypoints2, confidence,iter_num):
    Max_num = 0
    good_X = np.zeros([3,3])
    inlier_points = []
    for i in range(iter_num):
        random_points_set = randSeed(matches)
        x1,x2 = PointCoordinates(random_points_set, keypoints1, keypoints2)
        X = compute_fundamental(x1,x2)
        num, ransac_good = inlier(X, matches, keypoints1, keypoints2, confidence)
        if num > Max_num:
            Max_num = num
            good_X = X
            inlier_points = ransac_good
    return Max_num, good_X, inlier_points

def residual(inlier_points,keypoints1,keypoints2):
    x1, x2 = PointCoordinates(inlier_points, keypoints1, keypoints2)
    err = np.zeros([3,1])
    for i in range(len(x2)):
        err += np.sqrt((x2[i].reshape(-1,1)-x1[i].reshape(-1,1))**2)
    return err/reversed_match_num

########## Transformation #########################
Max_num, good_X, inlier_points = ransac(m12_reserved, kp1, kp2, 0.01,100)
residual = residual(inlier_points, kp1, kp2)
rows,cols,ch = image1.shape
M = np.array([good_X[0],good_X[1]])
dstAffine = cv2.warpAffine(image1, M, (cols,rows))
dstPerspective = cv2.warpPerspective(image1, good_X, (cols,rows))

########## Score #################################
def score(matches,keypoints1,keypoints2,X):
    x1, x2 = PointCoordinates(matches, keypoints1, keypoints2)
    err = np.zeros([3,1])
    for i in range(len(x2)):
        err += np.sqrt((x2[i].reshape(-1,1)-X.dot(x1[i].reshape(-1,1)))**2)
    return err/reversed_match_num

score = score(m12_reserved, kp1, kp2,good_X)
print("score")
print(score)

########## show image ############################
plt.subplot(221)
plt.imshow(image1)
plt.subplot(222)
plt.imshow(image2)
plt.subplot(223)
plt.imshow(dstAffine)
plt.subplot(224)
plt.imshow(dstPerspective)
plt.show()

########## draw  data #############################
# plt.plot(data1x,data1y,'b-',label='harris patch size')
# plt.title('harris corners')
# plt.xlabel('sigma')
# plt.ylabel('corners number')
# plt.legend()
# plt.show()
