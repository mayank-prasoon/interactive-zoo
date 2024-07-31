import cv2
import numpy as np
import math

# Define the range of the color you want to track in HSV
# Here, we define the range for the color green
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Get the default camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# reference pont to compare it against
reference_point = (frame_width // 2, frame_height // 2)

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, track the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        
        # Draw a rectangle around the largest contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If contours are found, track the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the moments of the largest contour
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            # Calculate the centroid
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Draw the center of the contour
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Calculate the distance from the reference point to the center of the contour
            distance = math.sqrt((center_x - reference_point[0]) ** 2 + (center_y - reference_point[1]) ** 2)
            
            # Print the coordinates of the center and the distance
            print(f"Center: ({center_x}, {center_y}), Distance: {distance:.2f}")
            
            # Draw the distance on the frame
            cv2.putText(frame, f"Dist: {distance:.2f}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw a rectangle around the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # # Optionally, you can draw the center of the contour
        # center_x = x + w // 2
        # center_y = y + h // 2
        # cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        cv2.circle(frame, reference_point, 5, (0, 0, 255), -1)

        # Print the coordinates of the center
        print(f"Center: ({center_x}, {center_y})")
    
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
