# All functions related to feature detection and matching

'''
TODO:
- Add feature detection
- Add feature matching to automate the point selection process
- Filter out false matches
- Maybe also here make it so that you can detect which image goes where without being given an order
'''

import numpy as np
import cv2
# import matplotlib.pyplot as plt


def get_match_pairs(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)
    good_matches = matches[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    return pts1, pts2, good_matches