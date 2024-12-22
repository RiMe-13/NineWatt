from PIL import Image
import numpy as np
import cv2

def find_max_circle_in_mask(mask):
    """
    Find the largest circle that fits entirely within the white region of the mask.
    
    Parameters:
        mask (numpy.ndarray): Binary mask image (0 = black, 255 = white).
    
    Returns:
        max_diameter (int): Maximum diameter of the circle in pixels.
        max_location (tuple): Coordinates of the circle's center (row, col).
    """
    # Perform distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Find the maximum radius and its location
    max_radius = np.max(dist_transform)
    max_location = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    
    # Convert radius to diameter
    max_diameter = int(max_radius * 2)
    
    return max_diameter, max_location

def draw_circle_on_mask(mask, center, radius):
    """
    Draw a circle on the mask for visualization purposes.
    
    Parameters:
        mask (numpy.ndarray): Binary mask image (0 = black, 255 = white).
        center (tuple): Coordinates of the circle's center (row, col).
        radius (int): Radius of the circle in pixels.
    
    Returns:
        result_image (numpy.ndarray): Mask image with the circle drawn.
    """
    # Convert grayscale mask to color image
    result_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw the circle
    cv2.circle(result_image, (center[1], center[0]), radius, (0, 0, 255), 2)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image

def draw_circle_on_original_image(original_image, center, radius, diameter):
    """
    Draw a circle on the original image for visualization purposes.
    
    Parameters:
        original_image (numpy.ndarray): Original image (color or grayscale).
        center (tuple): Coordinates of the circle's center (row, col).
        radius (int): Radius of the circle in pixels.
    
    Returns:
        result_image (numpy.ndarray): Original image with the circle drawn.
    """
    original_image = np.array(original_image)
    # Ensure original image is in color
    if len(original_image.shape) == 2:  # Grayscale image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    thickness = 4
    # Draw the circle on the original image
    cv2.circle(original_image, (center[1], center[0]), radius, (0, 255, 0), thickness)  # 초록색 원

    if diameter >= 16.97:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = radius / 50
        text_size = cv2.getTextSize('V', font, font_scale, thickness)[0]
        text_x = center[1] - text_size[0] // 2
        text_y = center[0] + text_size[1] // 2
        cv2.putText(original_image, 'V', (text_x, text_y), font, font_scale, (255, 255, 255), thickness)  # White 'V'
        
        return original_image
    else :
        return original_image


def find_max_circle(mask_image):
    # Convert PIL image to NumPy array
    mask_image = mask_image.convert('L')
    mask = np.array(mask_image)

    # Find the largest circle
    max_diameter, max_location = find_max_circle_in_mask(mask)

    # Draw the circle on the mask image
    result_image = draw_circle_on_mask(mask, max_location, max_diameter // 2)

    # Convert result back to PIL image for saving or display
    result_pil = Image.fromarray(result_image)

    # Save or display the result
    print(f"Maximum Diameter: {max_diameter}px, Center: {max_location}")
    return result_pil, max_diameter, max_location

import numpy as np
import cv2

def bring_evtol(original_image, center, radius):
    original_image = np.array(original_image)

    # Ensure original image is in color
    if len(original_image.shape) == 2:  # Grayscale image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    evtol_path = "./images/evtol.png"
    evtol_image = cv2.imread(evtol_path, cv2.IMREAD_UNCHANGED)
    
    if evtol_image is None:
        raise FileNotFoundError(f"Cannot find or open the evtol image at {evtol_path}")
    
    if evtol_image.shape[2] < 4:
        raise ValueError("evtol.png does not have an alpha channel.")
    
    # Extract the alpha channel and BGR channels from the evtol image
    alpha_channel = evtol_image[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]
    bgr_channels = evtol_image[:, :, :3]
    
    # Resize the evtol image to fit the specified radius (width = 2 * radius)
    scaling_factor = (2 * radius) / evtol_image.shape[1]  # Resize based on width
    new_size = (int(evtol_image.shape[1] * scaling_factor), int(evtol_image.shape[0] * scaling_factor))
    resized_bgr = cv2.resize(bgr_channels, new_size, interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha_channel, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Calculate the placement coordinates
    center_y, center_x = center  # Assuming center is (y, x)
    top_left_x = int(center_x - new_size[0] // 2)
    top_left_y = int(center_y - new_size[1] // 2)
    bottom_right_x = top_left_x + new_size[0]
    bottom_right_y = top_left_y + new_size[1]
    
    # Ensure coordinates are within bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(original_image.shape[1], bottom_right_x)
    bottom_right_y = min(original_image.shape[0], bottom_right_y)
    
    # Compute the regions where the evtol will be placed
    overlay_width = bottom_right_x - top_left_x
    overlay_height = bottom_right_y - top_left_y
    
    if overlay_width <= 0 or overlay_height <= 0:
        raise ValueError("The evtol image is outside the bounds of the original image.")
    
    # Crop the resized images if they exceed the original image boundaries
    resized_bgr = resized_bgr[:overlay_height, :overlay_width]
    resized_alpha = resized_alpha[:overlay_height, :overlay_width]
    
    # Extract the region of interest (ROI) from the original image
    roi = original_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Ensure the ROI and overlay have the same dimensions
    if roi.shape[0] != resized_bgr.shape[0] or roi.shape[1] != resized_bgr.shape[1]:
        raise ValueError("Mismatch in ROI and overlay dimensions.")
    
    # Perform alpha blending using vectorized operations
    resized_alpha = resized_alpha[:, :, np.newaxis]  # Make it (H, W, 1)
    blended = resized_alpha * resized_bgr + (1 - resized_alpha) * roi
    blended = blended.astype(original_image.dtype)
    
    # Replace the ROI on the original image with the blended image
    original_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = blended
    
    return original_image
