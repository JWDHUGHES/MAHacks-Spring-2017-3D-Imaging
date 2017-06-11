import cv2
import numpy as np
import sys
import math

def HLs(dst, cdst):
    lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
    if lines is not None:
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)


def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    return new_img

def fuck_this(grayObj, graySrc):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    objKeypoints, objDescrpt = orb.detectAndCompute(grayObj, None)
    sceneKeypoints, sceneDescrpt = orb.detectAndCompute(graySrc, None)

    matches = matcher.match(objDescrpt, sceneDescrpt)

    minDist = 100
    maxDist = 0

    for i in range(objDescrpt.size):
        dist = matches[i].distance
        if dist < minDist:
            minDist = dist
        if dist > maxDist:
            maxDist = dist

    goodMatches = []

    for i in range(objDescrpt.size):
        if matches[i].distance < 3*minDist:
            goodMatches.add(matches[i])

    imgMatches = drawMatches(obj, objKeypoints, grayObj, 
        graySrc, goodMatches)


    objPoints = []
    scenePoints = []

    for i in range(goodMatches.size):
        objPoints.add(objKeypoints[goodMatches[i].queryIdx].pt)
        scenePoints.add(sceneKeypoints[goodMatches[i].trainIdx].pt)

    return imgMatches

cap = cv2.VideoCapture(1)
welp = 1
median = 1
filter = 0

if __name__ == '__main__':
    while(True):
        # Capture frame-by-frame
        ret, src = cap.read()
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rawdst = cv2.medianBlur(src, median)
        rawdst2 = cv2.bilateralFilter(rawdst, welp, welp*2, welp/2)
        dst = cv2.Canny(rawdst2, 0, 170)
        cdst = cv2.cvtColor(255 - dst, cv2.COLOR_GRAY2BGR)

        filter = np.clip(cdst + filter, 0, 255)

        #new = cv2.distanceTransform(255 - dst, cv2.DIST_L2, 3)
        #new = cv2.normalize(new, 0.0, 1.0, cv2.NORM_MINMAX)
        obj = cv2.imread("sprite.jpg")
        grayObj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        minHessian = 400

        # if graySrc is not None and grayObj is not None:
        #    imgMatches = fuck_this(graySrc, grayObj)

       



        if ret:
            #HLs(dst, cdst)qqqqqaQq
            cv2.imshow("one", filter)
            cv2.imshow("mix", np.clip(filter + src, 0, 255))
            cv2.imshow("two", src)
            cv2.imshow("three", dst)
            #cv2.imshow("image", imgMatches)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('m'):
            median += 2
        if cv2.waitKey(1) & 0xFF == ord('n'):
            median -= 2    
        if cv2.waitKey(1) & 0xFF == ord('a'):
            welp += 1
        if cv2.waitKey(1) & 0xFF == ord('s'):
            welp -= 1
        if cv2.waitKey(1) & 0xFF == ord('r'):
            filter = 0

cap.release()
cv2.destroyAllWindows()
