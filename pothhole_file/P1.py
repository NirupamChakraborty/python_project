import cv2

# Test indices from 0 to 10 (adjust range as needed)
for index in range(10):
    cap = cv2.VideoCapture(index)  # Try opening the camera at this index
    if cap.isOpened():
        print(f"Camera found at index {index}")
        cap.release()  # Release the camera
    else:
        print(f"No camera found at index {index}")
