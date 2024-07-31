import cv2
import numpy as np
import math
import pyaudio

# Define the range of the green color in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Get the default camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =====================================================================================

# # Calculate the size of the 1:1 frame
# square_size = min(frame_width, frame_height)

# # Calculate the top-left corner of the square frame
# top_left_x = (frame_width - square_size) // 2
# top_left_y = (frame_height - square_size) // 2

# =====================================================================================

# Calculate the top-left corner of the square frame
top_left_x = (frame_width) // 2
top_left_y = (frame_height) // 2

elephant = (frame_width // 4, frame_height // 4)
birds    = (3 * frame_width // 4, frame_height // 4)
parrots  = (frame_width // 4, 3 * frame_height // 4)
monkey   = (3 * frame_width // 4, 3 * frame_height // 4)

# Define multiple fixed reference points
reference_points = [
    elephant,
    birds,
    parrots,
    monkey    
]

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the green color within the square frame
    mask = cv2.inRange(hsv, lower_green, upper_green)

# ==============================================================================================

    # # Zero out the areas outside the square frame using numpy slicing
    # mask[:top_left_y, :] = 0
    # mask[top_left_y+square_size:, :] = 0
    # mask[:, :top_left_x] = 0
    # mask[:, top_left_x+square_size:] = 0

# ==============================================================================================
    
    # Zero out the areas outside the square frame using numpy slicing
    mask[:top_left_y, :] = 0
    mask[top_left_y, :] = 0
    mask[:, :top_left_x] = 0
    mask[:, top_left_x] = 0

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, track the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the moments of the largest contour
        M = cv2.moments(largest_contour)
        
        # maths !!! alert !!!
        if M["m00"] != 0:
            # Calculate the centroid
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # This is more or less a point 
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Calculate and display the distance from each reference point to the center of the contour
            for point in reference_points:
                distance = math.sqrt((center_x - point[0]) ** 2 + (center_y - point[1]) ** 2)
                
                # Print the coordinates of the center and the distance
                print(f"Reference Point: {point}, Center: ({center_x}, {center_y}), Distance: {distance:.2f}")
                
                # Draw the distance on the frame
                cv2.putText(frame, f"Dist: {distance:.2f}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw a rectangle around the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw the 1:1 frame on the video feed

    # ============================================================================================================================
    
    # cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + square_size, top_left_y + square_size), (255, 0, 0), 2)
    
    # ============================================================================================================================

    cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x, top_left_y), (255, 0, 0), 2)
    
    # Draw the reference points
    for point in reference_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    
    # Display the frame and the mask
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
