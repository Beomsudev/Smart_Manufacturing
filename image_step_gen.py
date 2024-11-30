import cv2
import numpy as np
import os

# === 붉은색 부품 감지 및 저장 함수 ===
def detect_shunts_with_visualization(image, output_dir, image_name):
    """Detect shunts (red components) on the board, visualize the process, and save each step."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Save the original image
    cv2.imwrite(f"{output_dir}/{image_name}_step1_original.jpg", image)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Create a mask for red colors
    lower_red1 = np.array([0, 50, 50])  # Dark red range
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Bright red range
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    cv2.imwrite(f"{output_dir}/{image_name}_step2_red_mask.jpg", red_mask)

    # Step 3: Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{output_dir}/{image_name}_step3_cleaned_mask.jpg", red_mask)

    # Step 4: Find contours and detect shunts
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shunt_positions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(contour)
            shunt_positions.append((x, y))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle

    # Step 5: Annotate the image with shunt count
    annotated_image = image.copy()
    cv2.putText(annotated_image, f"Shunts: {len(shunt_positions)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/{image_name}_step4_annotated.jpg", annotated_image)

    return shunt_positions, annotated_image

# === 이미지 비교 및 저장 ===
def compare_images_with_steps(image_path_1, image_path_2, output_dir="output"):
    """Compare the shunts detected in two board images with step-by-step visualization and save the process."""
    # Load the two board images
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    if image1 is None or image2 is None:
        print("Error: Unable to load one or both images.")
        return

    # Process the first image
    print("Processing Image 1...")
    shunts1, annotated_image1 = detect_shunts_with_visualization(image1, output_dir, "Image1")

    # Process the second image
    print("Processing Image 2...")
    shunts2, annotated_image2 = detect_shunts_with_visualization(image2, output_dir, "Image2")

    # Combine the final annotated images for side-by-side comparison
    combined_height = max(annotated_image1.shape[0], annotated_image2.shape[0])
    combined_width = annotated_image1.shape[1] + annotated_image2.shape[1]
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_image[:annotated_image1.shape[0], :annotated_image1.shape[1]] = annotated_image1
    combined_image[:annotated_image2.shape[0], annotated_image1.shape[1]:] = annotated_image2

    # Save the combined image
    combined_output_path = f"{output_dir}/combined_result.jpg"
    cv2.imwrite(combined_output_path, combined_image)
    print(f"Combined result saved to: {combined_output_path}")

    # Display comparison results
    print(f"Shunts detected in Image 1: {len(shunts1)}")
    print(f"Shunts detected in Image 2: {len(shunts2)}")

# === 실행 ===
image_path_1 = "image/T1.jpg"  # Replace with the path to the first image
image_path_2 = "image/T2.jpg"  # Replace with the path to the second image
output_directory = "output_steps"  # Directory to save the outputs

# Run the comparison with visualization and save steps
compare_images_with_steps(image_path_1, image_path_2, output_dir=output_directory)
