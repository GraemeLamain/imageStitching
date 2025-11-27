# All Code related to image stitching and panorama creation
# Graeme Lamain - 100873910

import numpy as np
import cv2

def chain_homographies(valid_matches, key_points):
    '''
    Calculates the homography matrices for each image pair which has enough unique matching features.
    Uses RANSAC to filter out outliers (matches that do not fit the dominant plane)

    Input: valid_matches - Dictionary of matching features between image pairs
            key_points - Dictionary from get_key_points
    Output: H_matrices - Dictionary of 3x3 Homography matrices for each pair of images
    '''

    H_matrices = {}

    # Loop through each image pair with valid matches
    # If there are at least 10 valid matches, get homography between them
    for (i, j), matches in valid_matches.items():
        # Check that there are at least 10 valid matches
        if len(matches) < 10:
            print(f"Not enough matches ({len(matches)}) for pair {i} & {j}")
            continue

        # Get points
        print(f"Getting key points")
        kp_i = key_points[i]['keypoints']
        kp_j = key_points[j]['keypoints']

        # Grab only best matches
        # queryIdx is index in image i, trainIdx is image j
        print(f"Grab only best matches")
        src_pts = np.float32([kp_i[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_j[m.trainIdx].pt for m in matches])
        
        # RANSAC
        # Grabs a random sample of points and iterates to estimate a 3x3 homography matrix
        # It finds the matrix with the most inliers (agree to plane) and ignores the outliers for this matrix
        print(f"Starting RANSAC for Homography Estimation")
        H = ransac(src_pts, dst_pts, threshold=3.0, max_iters=500)

        if H is None:
            print("Homography failed.")
            continue

        if i not in H_matrices: H_matrices[i] = {}
        if j not in H_matrices: H_matrices[j] = {}

        # Store Homography I to J
        H_matrices[i][j] = H
        # Store inverse so we can go in either direction if we need to
        try:
            H_inv = np.linalg.inv(H)
            H_matrices[j][i] = H_inv
        except:
            print(f"Could not inverse matrix for {i}->{j}")

    return H_matrices

def global_homography(H_matrices, num_images):
    '''
    Chooses the middle image as our anchor point. Calculates the homography for every matrix to be mapped into the anchor space.
    This may require chaining (matrix multiplication) of multiple matrices in order to ensure all images are warped to the correct system.
    Creates a map to get from every image we have to the anchor image and stitch properly regardless of the order of images.

    Input: H_matrices - Homographies for image pairs
            num_images - Number of images user submitted
    Output: global_H - Dictionary mapping each image to the Anchor image space
    '''

    global_H = {}

    # Choose the middle image as our anchor point
    # This should limit distortion on numbers of images (there is better ways but much more complicated)
    anchor_idx = num_images //2

    # The anchor image doesnt need to change so use identity matrix
    global_H[anchor_idx] = np.eye(3)

    # Have to do BFS Queue because we do not know the order that the images were uploaded in
    # Traverse the graph starting from the anchor and go level by level (FIFO structure)
    # This is what allows us to handle unordered images as it can find the homography from any location
    queue = [(anchor_idx, np.eye(3))]
    visited = set([anchor_idx])

    print(f"Starting BFS Queue")
    while queue:
        current_idx, current_H_to_anchor = queue.pop(0)

        # Look at neighbors of current image
        if current_idx in H_matrices:
            for neighbor_idx in H_matrices[current_idx]:
                if neighbor_idx not in visited:
                    H_neighbor_to_current = H_matrices[neighbor_idx][current_idx]


                    # Chain Homographies
                    # Multiply the matricies together in order to combine them
                    H_neighbor_to_anchor = np.dot(current_H_to_anchor, H_neighbor_to_current)

                    global_H[neighbor_idx] = H_neighbor_to_anchor

                    visited.add(neighbor_idx)
                    queue.append((neighbor_idx, H_neighbor_to_anchor))
    return global_H

def calculate_canvas(global_H, key_points):
    '''
    Take the global homographies and map the corners of each image to the anchor space.
    Use this to find the maximum and minimum values for each axis to set the width and height of the canvas to stitch onto.
    
    Inputs: global_H - Dictionary mapping each image to the anchor image
            key_points - Dictionary from get_key_points
    Outputs: canvas_h, canvas_w - Dimensions of canvas for stitching
            offset - Shift for each axis to ensure everything fits on the canvas starting at (0, 0)
            valid_indicies - List of images which did not have exploding homographies (failed ones basically)
    '''

    corners = []
    valid_indices = []
    print(f"Starting Canvas Calculations")
    for idx, H in global_H.items():
        print(f"H for idx={idx}: {H}")

        img = key_points[idx]['image']
        h, w = img.shape[:2]

        # Get 4 corners of the original image (reshaped for cv2 reasons)
        corns = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1, 1, 2)

        # Transform using global H to map to the anchor space
        print(f"Transforming and appending corners")
        transformed = cv2.perspectiveTransform(corns, H)

        t_min = transformed.min(axis=0).ravel()
        t_max = transformed.max(axis=0).ravel()
                
        width = t_max[0] - t_min[0]
        height = t_max[1] - t_min[1]

        # Filters out images that are huge to prevent memory issues (happened in testing and went to like 100gb so this has to be here)
        # If the image is too big, likely means the homography exploded (failed basically)
        if width > 30000 or height > 30000:
            print(f"Image {idx} exploded from homography. W:{width}, H:{height}. Ignoring this image moving forward.")
            continue
        corners.append(transformed)
        valid_indices.append(idx)
    
    # Concatenate to find extremes of all images
    corners = np.concatenate(corners, axis=0)

    # Find min/max coordiantes for our canvas
    print(f"Get Extremes for corners")
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)

    padding = 50

    # Calculate offset for canvas
    # Use negative x & y min since they could be negative values
    offset = [-x_min + padding, -y_min + padding]

    canvas_w = x_max - x_min + (2 * padding)
    canvas_h = y_max - y_min + (2 * padding)

    return canvas_h, canvas_w, offset, valid_indices

def warp_and_stitch(global_H, key_points):
    '''
    Does the warping and stitching of each image onto the canvas.
    Calculates the canvas size to fit all images.
    Shifts all of the homographies by an offset/translation matrix so that we have no negative pixel values
    Warps each image and pasts them onto the canvas

    Inputs: global_H - Dictionary mapping each image to the anchor image
            key_points - Dictionary from get_key_points
    Outputs: panorama - Final panorama of all valid images stitched together
    '''

    # Calculate canvas dimensions to fit all images
    canvas_h, canvas_w, offset, valid_indices = calculate_canvas(global_H, key_points)

    # Create translation matrix
    T = np.array([
        [1, 0, offset[0]],
        [0, 1, offset[1]],
        [0, 0, 1]
    ], dtype=float)

    # Create blank canvas
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Warp all images
    print(f"Starting warps")
    for idx in valid_indices:
        H_to_1 = global_H[idx]
        img = key_points[idx]['image']

        # Shift homography by combining the translation with the global homography
        shifted_H = np.dot(T, H_to_1)

        # Warp the image
        # Using the vectorized version of the function since its alot faster than the original slow double for loop lol
        print(f"Warping for idx: {idx}")
        # warped_img = warp_img(img, shifted_H, [canvas_h, canvas_w])
        warped_img = warp_fast(img, shifted_H, [canvas_h, canvas_w])

        # Find where the warped image has valid data, the sum here should be greater than 0 or else its not there
        print(f"Masking idx: {idx}")
        mask = (np.sum(warped_img, axis=2) > 0)
        
        # Copy and paste thsoe pixels onto the panorama canvas
        print(f"Pasting idx: {idx}")
        panorama[mask] = warped_img[mask]
    
    return panorama

def ransac(src_pts, dst_pts, threshold=5.0, max_iters=500):
    '''
    Random Sample Consensus (RANSAC)
    Takes 4 random points (4 so we can solve for the 8 degrees of freedom we spoke of in class).
    Tries to estimate a homography matrix that fits as many points as it can.
    Takes the best matrix, i.e. matrix where the most points fit the plane
    Ignores any outlier points that exist for the estimation.

    Inputs: src_pts - Key points on the original image
            dst_pts - Key points on the image we want to map to
            threshold - Error for our points from the plane (how far are they from the plane)
            max_iters - Maximum number of iterations of the estimation
    Outputs: best_H - The best estimation of the homography mapping src_pts to dst_pts
    '''

    N = src_pts.shape[0]

    # We need at least 4 points to solve the homography matrix
    if N < 4:
        return None
    
    best_H = None
    max_inliers = 0
    best_inliers_mask = None

    # Need Homogeneous coordinates
    src_h = np.hstack((src_pts, np.ones((N, 1))))

    for i in range(max_iters):
        # Randomly sample 4 points
        idx = np.random.choice(N, 4, replace=False)

        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # Compute an estimation of a homography for these points
        try:
            H_candidate = homography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue

        # Apply the homography to every point in our original image
        # This is the vectorized way to do it and saves us a for loop
        # Transposes are just so that we can properly multiply
        projected = np.dot(H_candidate, src_h.T).T

        # Convert them back into cartesian coordinates by dividing by z
        # We avoid a division by 0 error by adding a super small epsilon
        # This means its a direction vector since its at infinity, 
        # but the point will be ignored anyways since its more than the threshold
        epsilon = 1e-10
        projected_x = projected[:, 0] / (projected[:, 2] + epsilon)
        projected_y = projected[:, 1] / (projected[:, 2] + epsilon)

        # Stack back into (N, 2)
        projected_pts = np.column_stack((projected_x, projected_y))

        # Calculate Error (Euclidean distance from projected and actual points)
        errors = np.linalg.norm(dst_pts - projected_pts, axis=1)

        # Count Inliers
        # Points where error is less than threshold
        current_inliers_mask = errors < threshold
        num_inliers = np.sum(current_inliers_mask)

        # Find the best model and only incude the inliers from them
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H_candidate
            best_inliers_mask = current_inliers_mask
    
    print(f"RANSAC finished. Best model has {max_inliers}/{N} inliers.")

    # Re-calculate H using ALL the inliers found (not just the random 4).
    # This gives a much more stable result.

    # To make the estimate just a bit better, we can recalculate a final homography using
    # all of the inlier points rather than just 4 random ones. Increases the accuracy of our estimation
    if max_inliers > 4:
        final_src = src_pts[best_inliers_mask]
        final_dst = dst_pts[best_inliers_mask]
        best_H = homography(final_src, final_dst)

    return best_H

def homography(src_pts, dst_pts):
    '''
    Calculates a 3x3 Homography matrix using the Direct Linear Transform (DLT) algorithm.
    Want to map points from the source image into the coordinate space of the destination image.
    We set up a system of linear equations and solve Ah = 0

    Inputs: src_pts - Key points on the original image
            dst_pts - Key points on the image we want to map to
    Outputs: a - Homography estimation
    
    '''
    N = src_pts.shape[0]

    H = []

    src_array = np.asarray(src_pts)
    dst_array = np.asarray(dst_pts)

    # Construct A matrix for Ah = 0
    # Matrix should be (2*N x 9)
    for n in range(N):
        # X-coordinate equation rows
        # [-x, -y, -1, 0, 0, 0]
        src = src_array[n]
        H.append(-src[0])   # -X
        H.append(-src[1])   # -Y
        H.append(-1)
        H.append(0)
        H.append(0)
        H.append(0)
    
    H = np.asarray(H)
    H1 = H.reshape(2*N, 3)

    # Y coordinate equation rows, do alternately cause thats what the algorithm wants
    H2 = np.zeros([2*N, 3], dtype=int)
    for i in range(0, 2*N, 2):
        H2[i:i+2,0:i+3] = np.flip(H1[i:i+2,0:i+3], axis=0)

    H2 = np.asarray(H2)
    H3 = np.concatenate((H1, H2), axis=1)

    # Fill final columns for x' and y' (destination coords)
    H4 = []
    for n in range(N):
        src = src_array[n]
        dst = dst_array[n]
        H4.append(src[0]*dst[0])        # x * x'
        H4.append(src[1]*dst[0])        # y * x'
        H4.append(dst[0])               # 1 * x'
        H4.append(src[0]*dst[1])        # x * y'
        H4.append(src[1]*dst[1])        # y * y'
        H4.append(dst[1])               # 1 * y'
    
    H4 = np.asarray(H4)
    H4 = H4.reshape(2*N, 3)

    H5 = np.concatenate((H3, H4), axis=1)

    # Now we have the equation Ah = 0, need to solve though

    # Solve the system using SVD or Eigen decomposition
    # Solve using A^T A
    H8 = np.matmul(np.transpose(H5), H5)        # Calculate AT A

    # Solution v is the eigenvector corresponding to the smallest eigen value w
    # Want smallest eigenvalue since that is the error in our estimation
    w, v = np.linalg.eig(H8)
    # print(f"w: {w}, v:{v}")
    min = w.min()

    # print(f"min: {min}")
    # Set up the estimation of our homography as a vector
    for i in range(len(w)):
        if w[i] == min:
            a = v[:, i]
    
    # Reshape into a 3x3 matrix and normalize so the last value is a 1
    a = np.asarray(a)
    a = a.reshape(3,3)
    a = a/a[2,2]

    return a

def warp_img(src_img, H, dst_img_size):
    '''
    OLDER FUNCTION - I added an updated faster version of this function called warp_fast

    Warps an image using inverse mapping.
    Iterates through every pixel in the destination image and applies the inverse homography.
    This finds the corresponding source pixel and it copies the color over

    Slow because its just iterating over every single pixel possible.
    This was my function that worked so I wanted to get everything else done before I optimized it.
    '''

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

def warp_fast(src_img, H, dst_img_size):
    '''
    Fast image warping from source image into destination image space.

    Uses Vectorized implementation of inverse mapping.
    Uses numPy broadcasting rather than for loops to do all pixels at once rather than iterating through them 1 by 1.

    Inputs: src_img - Image to be warped
            H - Homography to go from the source image to the destination
            dst_img_size - Canvas dimensions of panorama
    Output: dst_img - Warped image
    
    '''

    h_dst, w_dst = dst_img_size
    h_src, w_src = src_img.shape[:2]

    # Create a grid of coordinate pairs in dst
    # np.indices returns indices in (row, col) order, which corresponds to (y, x)
    grid_y, grid_x = np.indices((h_dst, w_dst))

    # Flatten them to a list of points (N, 2)
    # We need shape (3, N) for matrix multiplication: [x, y, 1] rows

    # Flatten to a list of points (N, 2)
    # Need it to be in the shape (3, N) for matrix multiplication: [x, y, 1] then however many rows
    ones = np.ones_like(grid_x)
    dst_coords_homogeneous = np.stack([grid_x.ravel(), grid_y.ravel(), ones.ravel()])

    # Apply the inverse homography to all of the points at once
    H_inv = np.linalg.inv(H)
    src_coords_homogeneous = H_inv @ dst_coords_homogeneous

    # Convert to cartesian coordinates by dividing by z
    # Avoid divide by zero by adding a tiny epsilon
    z = src_coords_homogeneous[2, :]
    z[z == 0] = 1e-10 
    
    src_x = src_coords_homogeneous[0, :] / z
    src_y = src_coords_homogeneous[1, :] / z

    # Round to nearest integer to find pixel indices
    src_x = np.round(src_x).astype(int)
    src_y = np.round(src_y).astype(int)

    # Create a Mask for Valid Pixels, this lets us do boundary checks on the entire image at once
    # True if the source coordinate is inside the source image
    mask = (src_x >= 0) & (src_x < w_src) & (src_y >= 0) & (src_y < h_src)

    # Map Pixels 
    # Create destination array
    dst_img = np.zeros((h_dst, w_dst, 3), dtype=src_img.dtype)
    
    # Reshape destination indices to flat arrays to match the mask
    dst_y_flat = grid_y.ravel()
    dst_x_flat = grid_x.ravel()

    # Apply the mask, only process pixels that are within the image bounds
    valid_dst_y = dst_y_flat[mask]
    valid_dst_x = dst_x_flat[mask]
    valid_src_y = src_y[mask]
    valid_src_x = src_x[mask]

    # Use advanced indexing to copy all of the valid pixels from the source to the destination in one operation
    dst_img[valid_dst_y, valid_dst_x] = src_img[valid_src_y, valid_src_x]

    return dst_img