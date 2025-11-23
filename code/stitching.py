# All Code related to image stitching and panorama creation

'''
TODO:
- Make blending not use cv2 functions
- Optimize homography
- Optimate warping
- (Optional) Make it so you can detect which image goes where without being given an order
'''

import numpy as np
import cv2


# Calculates Homography and maps points from srcs to dsts
def homography2(src_pts, dst_pts):
    N = src_pts.shape[0]

    H = []

    src_array = np.asarray(src_pts)
    dst_array = np.asarray(dst_pts)

    # Construct A matrix for Ah = 0
    for n in range(N):
        src = src_array[n]
        H.append(-src[0])
        H.append(-src[1])
        H.append(-1)
        H.append(0)
        H.append(0)
        H.append(0)
    
    H = np.asarray(H)
    H1 = H.reshape(2*N, 3)

    # Build alternative rows for y coords
    H2 = np.zeros([2*N, 3], dtype=int)
    for i in range(0, 2*N, 2):
        H2[i:i+2,0:i+3] = np.flip(H1[i:i+2,0:i+3], axis=0)

    H2 = np.asarray(H2)
    H3 = np.concatenate((H1, H2), axis=1)

    # Fill final columns for x' and y'
    H4 = []
    for n in range(N):
        src = src_array[n]
        dst = dst_array[n]
        H4.append(src[0]*dst[0])
        H4.append(src[1]*dst[0])
        H4.append(dst[0])
        H4.append(src[0]*dst[1])
        H4.append(src[1]*dst[1])
        H4.append(dst[1])
    
    H4 = np.asarray(H4)
    H4 = H4.reshape(2*N, 3)

    H5 = np.concatenate((H3, H4), axis=1)
    # Solve using A^T A
    H8 = np.matmul(np.transpose(H5), H5)

    w, v = np.linalg.eig(H8)
    print(f"w: {w}, v:{v}")
    min = w.min()
    print(f"min: {min}")
    for i in range(len(w)):
        if w[i] == min:
            a = v[:, i]
    
    a = np.asarray(a)
    a = a.reshape(3,3)
    a = a/a[2,2] #normalize

    return a

# ----- Blending wasnt asked for the lab, so i used cv2 functions -----

# Downsample and make Guassian pyramid
def gaussian_pyramid(img, levels):
    gp = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp

# Create Laplacian Pyramid from Gaussian Pyramid
def laplacian_pyramid(gp):
    lp = [gp[-1]] # Start at smallest
    for i in range(len(gp) - 1, 0, -1):
        # Upscale image
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        g_prime = cv2.pyrUp(gp[i], dstsize=size)
        # Create laplacian level
        laplacian = cv2.subtract(gp[i-1], g_prime)
        lp.append(laplacian)
    return lp


# Blend two laplacian pyramids using a mask pyramid
def laplacian_blend(lp_a, lp_b, mask_pyramid):
    blended = []
    for la, lb, m in zip(lp_a, lp_b, mask_pyramid):
        #  Adjust mask dimensions if they drift slightly during pyrDown
        if m.shape[:2] != la.shape[:2]:
            m = cv2.resize(m, (la.shape[1], la.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Expand mask to 3 channels for multiplication
        if len(m.shape) == 2:
            m = np.stack([m, m, m], axis=2)

        # Formula: blend = A * (1-M) + B * M
        ls = la * (1.0 - m) + lb * m
        blended.append(ls)
    return blended

# Collapse pyramid back down to single image
def reconstruct(lp):
    img = lp[0]
    for i in range(1, len(lp)):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, lp[i])
    return img

def stitch_img(img1, img2, pts1, pts2):
    PAD_Y = 400
    PAD_X = 400

    # Pad img1 to make room for warped image
    img1_padded = np.pad(img1, pad_width=((PAD_Y,PAD_Y), (PAD_X,PAD_X), (0,0)), mode='constant')
    
    # Shift reference points by padding amount
    dst_pts_shifted = []
    for p in pts1:
        dst_pts_shifted.append([p[0]+PAD_X, p[1]+PAD_Y])

    src_pts = pts2
    dst_pts_shifted = np.float32(dst_pts_shifted)
    src_pts = np.float32(src_pts)
    
    # Calculate Homography
    H = homography2(src_pts, dst_pts_shifted)
    # print("PTS1: ", pts1)
    # print("PTS2: ", pts2)
    print("Homography: ", H)
    
    # Warp img2 into img1 space
    img2_warped = warp_img(img2, H, [img1_padded.shape[0], img1_padded.shape[1]]).astype(np.uint8)

    print("Warped Image Shape: ", img2_warped.shape)
    # mask = (np.sum(img2_warped, axis=2) > 0)

    # Create and process mask
    mask = (np.sum(img2_warped, axis=2) > 0).astype(np.float32)
    
    # Erode to remove edge artifacts from warping
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
    
    # Blur to make gradient when blending
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    
    # Laplacian Blending
    img1_f = img1_padded.astype(np.float32)
    img2_f = img2_warped.astype(np.float32)

    # Generate gaussian pyramids for both and the mask
    levels = 10
    gp_1 = gaussian_pyramid(img1_f, levels)
    gp_2 = gaussian_pyramid(img2_f, levels)
    gp_mask = gaussian_pyramid(mask.astype(np.float32), levels)
    
    # Generate laplacian pyramids for both images
    lp_1 = laplacian_pyramid(gp_1)
    lp_2 = laplacian_pyramid(gp_2)

    # Blend pyramids
    blended_pyramid = laplacian_blend(lp_1, lp_2, gp_mask[::-1])

    # Collapse pyramid to get final image
    result = reconstruct(blended_pyramid)

    # Clip values to valid 0-255 range and conver tto uint8
    out_img = np.clip(result, 0, 255).astype(np.uint8)

    return out_img

# Warps image using inverse mapping
def warp_img(src_img, H, dst_img_size):
    dst_img = np.zeros([dst_img_size[0], dst_img_size[1], 3])

    M = dst_img.shape[0]
    N = dst_img.shape[1]

    # Probably not the optimized way to do it but it works
    for i in range(N):      # x loop
        for j in range(M):  # y loop
            coords = [i, j, 1]
            coords = np.asarray(coords)
            coords = coords.transpose()

            # Inverse mapping
            H_inv = np.linalg.inv(H)
            new_pts = np.matmul(H_inv, coords)

            # Normalize homogenous coords
            src_x = round(new_pts[0]/new_pts[2])
            src_y = round(new_pts[1]/new_pts[2])

            # check boundaries
            if (src_y < 0 or src_x < 0) or (src_x > src_img.shape[1] or src_y > src_img.shape[0]):
                dst_img[j, i] = 0
            elif (src_y > 0 and src_x > 0) and (src_x < src_img.shape[1] and src_y < src_img.shape[0]):
                dst_img[j, i] = src_img[src_y, src_x]
    
    return dst_img[:M, :N]