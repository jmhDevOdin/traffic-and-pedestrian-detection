import cv2
from random import randrange

# Test samples
# car_image = 'sample_img.png'
sample = cv2.VideoCapture('sample_video.mp4')

# Pre-trained classifier based on haar cascade algorithm in openCV
car_detector = cv2.CascadeClassifier('car_detector.xml')
pedestrian_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Run for sample length (infinite for live video feeds)
while True:

    # Read current frame
    (read_successful, frame) = sample.read()

    # Error checking
    if (read_successful):
        # Convert to grayscale for improving detection
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars in sample
    cars = car_detector.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_detector.detectMultiScale(grayscale_frame)

    # Draw rectangles around cars in red
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangle around pedestrians in yellow
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display image with cars spotted
    cv2.imshow('Traffic & Pedestrian Detection - Press Q key to exit', frame)

# Stop autoclose so we can see
    key = cv2.waitKey(1)
    if key==11 or key==113:
        break

# Free system resources
sample.release()
cv2.destroyAllWindows()