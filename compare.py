import cv2
import numpy as np

def detect_shunts(image):
    """Detect shunts (red components) on the board."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Clean up the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the detected red regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shunt_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(contour)
            shunt_positions.append((x, y))
            # Draw rectangles around detected shunts for visualization
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return shunt_positions, image

def combine_images(image1, image2, output_path="result.jpg"):
    """Combine two images side by side and save the result."""
    # Ensure both images have the same height
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    combined_height = max(height1, height2)
    combined_width = width1 + width2

    # Create a blank image to hold both images
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place the first image on the left
    combined_image[:height1, :width1] = image1

    # Place the second image on the right
    combined_image[:height2, width1:width1 + width2] = image2

    # Save the combined image
    cv2.imwrite(output_path, combined_image)

def compare_images(image_path_1, image_path_2):
    """Compare the shunts detected in two board images."""
    # Load the two board images
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    if image1 is None or image2 is None:
        print("Error: Unable to load one or both images.")
        return

    # Detect shunts in both images
    shunts1, annotated_image1 = detect_shunts(image1)
    shunts2, annotated_image2 = detect_shunts(image2)

    # Add shunt count text to the images
    cv2.putText(annotated_image1, f"Shunts: {len(shunts1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_image2, f"Shunts: {len(shunts2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine the two annotated images into one
    combine_images(annotated_image1, annotated_image2, output_path="result.jpg")

    # Display comparison results
    print(f"Shunts detected in Image 1: {len(shunts1)}")
    print(f"Shunts detected in Image 2: {len(shunts2)}")

# Provide the paths to the images
image_path_1 = "image/T1.jpg"  # Replace with the path to the first image
image_path_2 = "image/T2.jpg"  # Replace with the path to the second image

# Run the comparison
compare_images(image_path_1, image_path_2)
