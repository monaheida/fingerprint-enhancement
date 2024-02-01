
# Based on the paper "Liveness detection for fingerprint scanners
# ... Bozaho Tan"
####################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

def computeAngle(dx, dy):
    angle = 0
    if (dx > 0 and dy >=0):
        angle = math.atan(dy / dx)
    elif (dx > 0 and dy < 0):
        angle = math.atan(dy / dx) + 2 * np.pi
    elif (dx < 0):
        angle = math.atan(dy / dx) + np.pi
    elif (dx == 0 and dy > 0):
        angle = np.pi / 2
    elif (dx == 0 and dy < 0):
        angle = 3 * np.pi / 2

    return angle

def orientation(orig_img, orientation, blk_size = 25):
    img = np.float32(orig_img) / 255.0

    half_blk_size = int(blk_size / 2)  # The center of the w x w block
    height, width = img.shape
        
    # compute X and Y gradients field
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    orientation = np.zeros([int(height / blk_size), int(width / blk_size)], dtype = float)

    # create color image to display the orientations
    color_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    for i in range(half_blk_size, height-half_blk_size, blk_size):
        for j in range(half_blk_size, width-half_blk_size, blk_size):
            #print (i, " ", j)
            sum_Vx = 0
            sum_Vy = 0
            for u in range(i-half_blk_size, i+half_blk_size):
                for v in range(j-half_blk_size, j+half_blk_size):
                    sum_Vx = sum_Vx + (2*grad_x[u][v] * grad_y[u][v])
                    sum_Vy = sum_Vy + ((grad_x[u][v] * grad_x[u][v]) - (grad_y[u][v] * grad_y[u][v]))

            angle = computeAngle(sum_Vy, sum_Vx) / 2 + np.pi / 2
            if angle >= np.pi:
                angle -= np.pi
            orientation[int(i / blk_size)][int(j / blk_size)] = angle

            # draw to color image            
            dx = math.cos(angle) * (half_blk_size - 1)
            dy = math.sin(angle) * (half_blk_size - 1)
            cv2.line(color_img, (int(j + dx), int(i + dy)), (int(j - dx), int(i - dy)), (0, 0, 255), 2)
            
    print(orientation)
    cv2.imshow("Sobel", color_img)
    cv2.waitKey(0)

def FFT(patch):
    pad = 16
    # pad the (stride) image with zeros
    patch_padded = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    # create output storage
    planes = [np.float32(patch_padded), np.zeros(patch_padded.shape, np.float32)]
    complexI = cv2.merge(planes)

    #do the fourier transform
    cv2.dft(complexI, complexI)

    cv2.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]

    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv2.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    
    #cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX) # Transform the matrix with float values into a

    # find max response
    minVal, maxVal, locMin, locMax = cv2.minMaxLoc(magI)

    # compute angle from location of max response
    #angle = (np.pi / 180) * cv2.fastAtan2(magI.shape[0] / 2 - locMax[1], locMax[0] - magI.shape[1] / 2) + np.pi / 2
    angle = np.arctan2(magI.shape[0] / 2 - locMax[0], locMax[1] - magI.shape[1] / 2)

    #cv2.imshow("spect", magI)
    #cv2.imshow("patch", patch)
    #cv2.waitKey(0)

    return angle, maxVal

def orientationFFT(orig_img, blk_size):
    img = np.float32(orig_img) / 255.0

    blur = cv2.GaussianBlur(img, (15, 15), 0)

    edge_image = img - blur
    
    #img_inv = 1 - img
    height, width = edge_image.shape

    half_blk_size = int(blk_size / 2)  # The center of the w x w block

    #cv2.imshow("ori", edge_image)
    #cv2.waitKey(0)

    # create color image to display the orientations
    color_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    for i in range(half_blk_size, height-half_blk_size, blk_size):  # rows
        for j in range(half_blk_size, width-half_blk_size, blk_size):   # columns

            levels = 1
            angles = []
            amplitudes = []
            for a in range(0, levels):

                # extract the patch
                patch = edge_image[(i - half_blk_size - a * 4):(i + half_blk_size - a * 4), (j - half_blk_size - a * 4):(j + half_blk_size - a * 4)]

                angle, amplitude = FFT(patch)
                angles.append(angle)
                amplitudes.append(amplitude)

            # find index of element with largest amplituse (response)
            idx = np.argmax(amplitudes)
            # draw the orientation into image
            if amplitudes[idx] > 1.5:
                dx = math.cos(angles[idx]) * (half_blk_size - 1)
                dy = math.sin(angles[idx]) * (half_blk_size - 1)
                cv2.line(color_img, (int(j + dx), int(i + dy)), (int(j - dx), int(i - dy)), (0, 0, 255), 2)

            #cv2.imshow("ori", color_img)
            #cv2.waitKey(0)

    cv2.imshow("FFT", color_img)
    cv2.waitKey(0)       

def gaborOrientations(orig_img, blk_size = 16, gabor_size = 12, n_orientations = 16):
    #create gabor kernels
    v_kernels = []
    for i in range(0, n_orientations):
        kernel = cv2.getGaborKernel((gabor_size, gabor_size), gabor_size, 
            np.pi / 2 + (np.pi / n_orientations) * i, 15, 0.5, 0, cv2.CV_32F)
        v_kernels.append(kernel)
     
    img = np.float32(orig_img) / 255.0
    height, width = img.shape
 
    #create inverse image: white = ridge
    img_inv = 1.0 - img
 
    #filter images using kernels
    v_responses = []
    for i in range(0, n_orientations):
        response = cv2.filter2D(img_inv, -1, v_kernels[i])
        v_responses.append(response)
 
    #create maximum response image
    max_response = np.zeros(img.shape, dtype = float)
    dense_orientation = np.zeros(img.shape, dtype = float)
 
    for i in range(0, height):        #rows
        for j in range(0, width):    #columns
             
            #find image id with best response
            best_id = 0
            best_response = v_responses[best_id][i, j]
            for a in range(0, len(v_responses)):
                if v_responses[a][i, j] > best_response:
                    best_response = v_responses[a][i, j]
                    best_id = a
 
            #store best angle and response
            max_response[i, j] = best_response
            dense_orientation[i, j] = (np.pi / n_orientations) * best_id
 
    #create color image to display the orientations
    color_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
 
    half_blk_size = int(blk_size / 2)
    orientation = np.zeros([int(height / blk_size), int(width / blk_size)], dtype = float)
    #build orientations for boxes
    for i in range(half_blk_size, height-half_blk_size, blk_size):
        for j in range(half_blk_size, width-half_blk_size, blk_size):
            #find maximum response in this window
            patch = max_response[i-half_blk_size:i+half_blk_size, j-half_blk_size:j+half_blk_size]
            minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(patch)
 
            if maxV < 22:
                continue
 
            #extract rotation from max response point
            angle = dense_orientation[i-half_blk_size + maxLoc[1], j-half_blk_size + maxLoc[0]]
            orientation[int(i / blk_size)][int(j / blk_size)] = angle
 
            dx = math.cos(angle) * (half_blk_size - 1)
            dy = math.sin(angle) * (half_blk_size - 1)
            cv2.line(color_img, (int(j + dx), int(i + dy)), (int(j - dx), int(i - dy)), (0,0, 255), 2)
 
    cv2.imshow("Gabor", color_img)
    cv2.waitKey(0)
 
    return orientation, max_response

if __name__ == "__main__":
    orig_img = cv2.imread('/home/sishel/Desktop/App/data/FP_00094.jpg', cv2.IMREAD_GRAYSCALE)
    orientation(orig_img, orientation, blk_size = 16)
    orientationFFT(orig_img, blk_size = 16)
    gaborOrientations(orig_img, blk_size=16, gabor_size = 12, n_orientations = 16)
