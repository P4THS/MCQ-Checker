#Qasim Naveed    21L-5231
#Rameez          21L-1775

import cv2
import numpy as np
import math

def inRange(image, lower_bound, upper_bound):
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
         
            is_within_range = True

            for k in range(c):
                if not (lower_bound[k] <= image[i, j, k] <= upper_bound[k]):
                    is_within_range = False
                    break 

            if is_within_range:
                mask[i, j] = 255  
            else:
                mask[i, j] = 0   
    
    return mask


def bitwise_not(mask):
    return 255 - mask


def convert_to_grayscale(image):
    B = image[:, :, 0]  
    G = image[:, :, 1]  
    R = image[:, :, 2]  
    
    gray_image = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)
    
    return gray_image


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    mean = size // 2  
   
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * math.pi * sigma**2)) * \
                           math.exp(-((i - mean)**2 + (j - mean)**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, kernel_size, sigma):
    h, w = image.shape
    pad = kernel_size // 2 
    blurred_image = np.zeros_like(image, dtype=np.float32)

    kernel = gaussian_kernel(kernel_size, sigma)

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            blurred_value = np.sum(region * kernel)
           
            blurred_image[i, j] = blurred_value

    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    
    return blurred_image


def adaptive_threshold(image, max_value, adaptive_method, threshold_type, block_size, C):
    h, w = image.shape
    pad = block_size // 2  
    
    output_image = np.zeros_like(image, dtype=np.uint8)
    
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            block = padded_image[i:i + block_size, j:j + block_size]
            
            if adaptive_method == 'mean':
                threshold = np.mean(block) - C
            elif adaptive_method == 'gaussian':
                kernel = np.exp(-((np.arange(block_size) - pad) ** 2) / (2 * (pad ** 2)))
                kernel = kernel / np.sum(kernel)
                weighted_mean = np.sum(block * kernel[:, None])  
                weighted_mean = np.sum(weighted_mean * kernel)  
                threshold = weighted_mean - C
            else:
                raise ValueError("Unknown adaptive method. Use 'mean' or 'gaussian'.")

            if threshold_type == 'binary_inv':
                if image[i, j] < threshold:
                    output_image[i, j] = max_value
                else:
                    output_image[i, j] = 0
            elif threshold_type == 'binary':
                if image[i, j] >= threshold:
                    output_image[i, j] = max_value
                else:
                    output_image[i, j] = 0
            else:
                raise ValueError("Unknown threshold type. Use 'binary_inv' or 'binary'.")
    
    return output_image

def detect_circles(image, dp=1.2, minDist=40, param1=50, param2=30, minRadius=15, maxRadius=30):

    edges = cv2.Canny(image, param1, param2)
    
    max_radius = maxRadius
    min_radius = minRadius
    rows, cols = edges.shape
    max_accumulator_value = int(np.sqrt(rows**2 + cols**2))  #
    
    accumulator = np.zeros((rows, cols, max_radius - min_radius + 1), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if edges[i, j] > 0:
                for radius in range(min_radius, max_radius + 1):
                    for angle in range(0, 360, 1): 
                        a = int(i - radius * np.cos(angle * np.pi / 180))
                        b = int(j - radius * np.sin(angle * np.pi / 180))
                        
                        if 0 <= a < rows and 0 <= b < cols:
                            accumulator[a, b, radius - min_radius] += 1

    detected_circles = []
    for i in range(rows):
        for j in range(cols):
            for radius_idx in range(accumulator.shape[2]):
                if accumulator[i, j, radius_idx] > param2:  
                    detected_circles.append((j, i, radius_idx + min_radius))
    
    final_circles = []
    for (x, y, r) in detected_circles:
        too_close = False
        for (cx, cy, cr) in final_circles:
            if np.sqrt((cx - x) ** 2 + (cy - y) ** 2) < minDist:
                too_close = True
                break
        if not too_close:
            final_circles.append((x, y, r))

    circles = np.array(final_circles, dtype=np.float32)
    
    circles = circles.reshape((1, circles.shape[0], 3))

    return circles

def apply_convolution(image, kernel):
    h, w = image.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output_image[i, j] = np.sum(region * kernel)

    return output_image

def gradient_magnitude_and_direction(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    direction[direction < 0] += 180 
    
    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    h, w = magnitude.shape
    output_image = np.zeros_like(magnitude)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]
            elif (22.5 <= angle < 67.5):
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]
            elif (67.5 <= angle < 112.5):
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            else:
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                output_image[i, j] = magnitude[i, j]
            else:
                output_image[i, j] = 0

    return output_image

def double_thresholding(image, low_threshold, high_threshold):
    h, w = image.shape
    strong_edges = np.zeros_like(image, dtype=np.uint8)
    weak_edges = np.zeros_like(image, dtype=np.uint8)

    strong_edges[image >= high_threshold] = 255
    weak_edges[(image >= low_threshold) & (image < high_threshold)] = 50

    return strong_edges, weak_edges

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    h, w = strong_edges.shape
    final_edges = np.copy(strong_edges)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak_edges[i, j] == 50:
                if np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                    final_edges[i, j] = 255
                else:
                    final_edges[i, j] = 0

    return final_edges

def canny_edge_detection(image, low_threshold, high_threshold, kernel_size=5, sigma=1.4):
    gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)
    blurred_image = apply_convolution(image, gaussian_kernel_matrix)
    magnitude, direction = gradient_magnitude_and_direction(blurred_image)
    suppressed_image = non_maximum_suppression(magnitude, direction)
    strong_edges, weak_edges = double_thresholding(suppressed_image, low_threshold, high_threshold)
    final_edges = edge_tracking_by_hysteresis(strong_edges, weak_edges)

    return final_edges


def bitwise_and(image1, image2, mask=None):
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same shape.")
    
    result = np.zeros_like(image1, dtype=np.uint8)

    for i in range(image1.shape[0]):  
        for j in range(image1.shape[1]):  
            if mask is not None and mask[i, j] == 255:
                result[i, j] = image1[i, j] & image2[i, j]
            elif mask is None:
                result[i, j] = image1[i, j] & image2[i, j]
    
    return result

def count_non_zero_pixels(image):
    count = 0
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i, j] != 0:  
                count += 1
    return count

def calculate_bubble_ratio(masked_bubble, r):
    
    non_zero_count = count_non_zero_pixels(masked_bubble)
    
    area_of_circle = np.pi * (r ** 2)
    
    ratio = non_zero_count / area_of_circle
    return ratio


image_path = 'filled10mcq.png'
input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])

white_mask = inRange(input_image, lower_white, upper_white)

non_white_image = bitwise_not(white_mask)

gray_image = convert_to_grayscale(input_image)

for i in range(gray_image.shape[0]):  
    for j in range(gray_image.shape[1]): 
        if non_white_image[i, j] == 255:
            gray_image[i, j] = 0

blurred_image = apply_gaussian_blur(gray_image, 5, 1)

thresh_image = adaptive_threshold(blurred_image, max_value=255, adaptive_method='mean', threshold_type='binary_inv', block_size=3, C=5)

# cv2_output = cv2.Canny(thresh_image, 50, 30)
# custom_output = canny_edge_detection(thresh_image, 50, 30)

# np.savetxt("cv2_canny_output.txt", cv2_output, fmt="%d")
# np.savetxt("custom_canny_output.txt", custom_output, fmt="%d")

circles = cv2.HoughCircles(thresh_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                           param1=50, param2=30, minRadius=15, maxRadius=30)


circle_indices = {'empty': [], 'filled': []}
output_image = input_image.copy()

if circles is not None:
    circles = np.round(circles[0, :]).astype('int')
    
    circles = sorted(circles, key=lambda c: (c[1] // 40, c[0]))

    for i, (x, y, r) in enumerate(circles):
        mask = np.zeros_like(gray_image)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        masked_bubble = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        fill_ratio = cv2.countNonZero(masked_bubble) / (np.pi * r**2)
        
        if fill_ratio > 0.2:
            circle_indices['empty'].append(i+1)
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 3)  
        else:
            circle_indices['filled'].append(i+1)
            cv2.circle(output_image, (x, y), r, (0, 0, 255), 3) 

output_path = 'detected_sorted.jpg'

cv2.imwrite(output_path, output_image)
cv2.imshow('Detected Bubbles Sorted', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Sorted Circle Indices:', circle_indices)

def grade_mcq_sheet(answer_key, filled_indices):
    marks = 0.0
    detailed_results = {}

    for question, correct_answer in enumerate(answer_key, 1):
        question_options = [
            (question - 1) * 4 + 1,
            (question - 1) * 4 + 2,
            (question - 1) * 4 + 3,
            (question - 1) * 4 + 4
        ]

        option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

        filled_options = [
            opt for opt in question_options 
            if opt in filled_indices['filled']
        ]
        
        filled_alphabets = [
            option_map[(opt - 1) % 4 + 1] for opt in filled_options
        ]
        
        print(f'{question}: {filled_alphabets}')
        
        if len(filled_options) > 1:
            marks -= 0.25
            detailed_results[question] = {
                'status': 'Multiple options',
                'penalty': -0.25,
                'filled_options': filled_alphabets
            }
        elif len(filled_options) == 1:
            correct_option_index = (question - 1) * 4 + ['A', 'B', 'C', 'D'].index(correct_answer) + 1
            
            if filled_options[0] == correct_option_index:
                marks += 1
                detailed_results[question] = {
                    'status': 'Correct',
                    'marks': 1,
                    'filled_option': filled_alphabets[0]
                }
            else:
                detailed_results[question] = {
                    'status': 'Incorrect',
                    'marks': 0,
                    'filled_option': filled_alphabets[0]
                }
        else:
            detailed_results[question] = {
                'status': 'Not attempted',
                'marks': 0
            }
    
    return {
        'total_marks': marks,
        'detailed_results': detailed_results
    }

answer_key = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']

filled_indices = {
    'filled': circle_indices['filled']
}

result = grade_mcq_sheet(answer_key, filled_indices)
print(result)