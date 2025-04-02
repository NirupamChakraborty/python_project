# THIS IS A COLLABORATIVE PROJECT
import cv2


# Open video file or use webcam (use 0 for webcam)
# video = cv2.VideoCapture(r'pothole1.mp4')  # Replace with your video file path
video = cv2.VideoCapture(0)  # Uncomment to use webcam

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video or cannot access video source.")
        break

    # Resize frame for consistent processing size
    frame = cv2.resize(frame, (400, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    # Edge detection using Canny
    edges = cv2.Canny(blur, 80, 150)


    # Apply a binary threshold
    _, threshold = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # # Edge detection using Canny
    # edges = cv2.Canny(blur, 80, 150)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter based on contour area to avoid small noise
            # Draw bounding rectangles for detected potholes
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Pothole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the processed frames
    
    # h = np.hstack((frame, cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR),cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR) ))
    # cv2.imshow("output", h)
    cv2.imshow("Original with Pothole Detection", frame)
    cv2.imshow("Thresholded Image", threshold)
    cv2.imshow("blur", blur)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()