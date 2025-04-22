# capture.py (Runs on Python 3.11)
import cv2
import numpy as np
from picamera2 import Picamera2

def capture_image():
    camera = Picamera2()
    camera.configure(camera.create_still_configuration(main={'size': (800, 800)}))
    camera.start()
    
    img = camera.capture_array()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img[100:780, 15:800]
    #img = img[20:700, 0:785]
    img = cv2.resize(img, (800, 800))
    img = np.rot90(img, 2)  # Rotate 180 degrees
    
    cv2.imwrite("frame.jpg", img)  # Save image for inference
    print("Image Captured")

if __name__ == "__main__":
    capture_image()
