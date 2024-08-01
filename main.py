import cv2
import numpy as np
import math
import animal
from pygame import mixer

# Define the range of the color you want to track in HSV
# Here, we define the range for the color green
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Get the default camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print(frame_width, frame_height)
# # reference pont to compare it against
# reference_point = (frame_width // 2, frame_height // 2)

# animal location
animal_locs = animal.animal_locs

# create mixer channel 
mixer.init()
mixer.set_num_channels(13)

# loop though and assign channel with the respective code base 
for idx, animal in enumerate(animal_locs):
    if animal[1]:
        mixer.Channel(idx).play(mixer.Sound(animal[1]))
    else:
        mixer.Channel(idx).play(mixer.Sound('assets/battle_theme_regular.mp3'))

    mixer.Channel(idx).set_volume(0.0)
    print(animal)


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
            
            # loop though the array to find the 
            for idx, animal in enumerate(animal_locs):
                max_distance = animal[2] # each animal have a different maximum length
                cv2.circle(frame, animal[0], 5, (0, 0, 255), -1) # draw a circle around the point

                # calculate the distance and based on the distance calculate the volume
                distance = math.sqrt((center_x - animal[0][0]) ** 2 + (center_y - animal[0][1]) ** 2)
                volume   = max(0, min(1, 1-(distance/max_distance))) # check for the min value 
                mixer.Channel(idx).set_volume(volume)
        
        # Draw a rectangle around the largest contour
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # frame = np.fliplr(frame)
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
