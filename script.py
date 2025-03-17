import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def align_image(image, reference):
    sift = cv2.SIFT_create()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray_image, None)
    kp2, des2 = sift.detectAndCompute(gray_reference, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("⚠️ Not enough keypoints for alignment!")
        return image  

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(image, matrix, (reference.shape[1], reference.shape[0]))

    return aligned


def extract_high_freq(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 5)
    return cv2.subtract(image, blurred)

def get_reference_image(blurry_images):
    def high_freq_content(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    high_freq_values = [high_freq_content(img) for img in blurry_images]

    return blurry_images[np.argmax(high_freq_values)]


def multi_frame_fusion(blurry_images):
    reference = get_reference_image(blurry_images)
    fused = np.zeros_like(reference, dtype=np.float32)

    for img in blurry_images:
        aligned_img = align_image(img, reference)
        high_freq = extract_high_freq(aligned_img)

        high_freq = cv2.warpPerspective(high_freq, np.eye(3), (reference.shape[1], reference.shape[0]))

        fused += high_freq

    fused = cv2.normalize(fused, None, 0, 1, cv2.NORM_MINMAX)#.astype(np.uint)

    return fused

blurry_images = os.listdir("clicked_dataset/nb_1")
blurry_images