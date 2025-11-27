# All functions related to feature detection and matching
# Graeme Lamain - 100873910

import cv2

def get_key_points(imgs):
    '''
    Use CV2 Scale Invarient Feature Transform (SIFT) to extract key features from each image and get descriptors for them.
    Creates a pyramid of images at different levels of gaussian blur and subtracts them from eachother (Difference of Gaussian).
    It does this to find pixels that stand out from the rest as being brighter or darker regardless of scale or rotation.

    Descriptors are 128-values which represent the gradient directions of the pixels surrounding the keypoint.

    Input: imgs - List of images
    Output: key_points - Dictionary of images with their key points and descriptors
    '''
    key_points = {}
    for i in range(len(imgs)):
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(imgs[i], None)
        key_points[i] = {
            "keypoints": kp,
            "descriptors": des,
            "image": imgs[i]
        }

    return key_points

def get_matches(imgs, key_points):
    '''
    Uses Brute Force Matching and K-Nearest Neighbors to match features between every pair of images.
    Uses Lowe's Ratio Test to ensure only unique matches are kept.

    Input: imgs - List of images
           key_points - Dictionary from get_key_points
    Output: matches - Dictionary where keys are tuples (i, j) and values are lists of DMatch objects.
    '''

    N = len(imgs)

    matches = {}

    # Attemps to match every descriptor in one image to every descriptor in another
    # Uses NORM_L2 as Euclidean distance to compare the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    print("Starting loop for matches")
    try:
        for i in range(N - 1):              # Stops before the last image
            for j in range(i + 1, N):       # Start just after I, goes to the end
                des1 = key_points[i]['descriptors']
                des2 = key_points[j]['descriptors']

                # Make sure there are descriptors in the lists
                # Need at least 2 in order to find the 2 best matches
                print(f"Checking for valid matches on {i} & {j}")
                if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                    # print(f"No Matches for pair {i} & {j}")
                    continue
                
                # Use KNN to find the top 2 best matches for each feature
                raw_matches = bf.knnMatch(des1, des2, k=2)
                good_matches = []

                # Filter out bad matches using Lowes Ratio Test
                # Gets ratio between distances of best match and second best match
                # If the ratio is lower than our threshold, then we accept it as a good match
                # This gets rid of false matches
                # If the second best match is no where near the best, then it is unique and we want to keep it

                # Use Lowe's Ratio test to filter out some of our matches
                # If the two best matches are very close in distance it means that the feature is not unique
                # since they are very similar to each other and could be a repeating pattern.
                # Only keep matches where the best match is much closer than the second. 
                for m, n in raw_matches:
                    if m.distance < 0.50 * n.distance:
                        good_matches.append(m)
                
                matches[(i, j)] = good_matches
        print(f"Finished getting Matches: {len(matches)}")   
    except Exception as e:
        print(f"Failed to get matches: {e}")

    return matches