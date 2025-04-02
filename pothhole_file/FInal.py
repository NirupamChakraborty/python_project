import cv2
import numpy as np

# Load YOLO network
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
yolo_classes = []
with open("coco.names", "r") as f:
    yolo_classes = f.read().strip().split("\n")

# Start video capture
cap = cv2.VideoCapture(0)  # Use a video file path for offline processing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    output_layers = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers)

    # Initialize lists for detected objects
    boxes, confidences, class_ids = [], [], []

    # Parse YOLO layer outputs
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.3:  # Confidence threshold
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to eliminate redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        for i in indices.flatten():  # Flatten indices for easy iteration
            x, y, w, h = boxes[i]
            label = str(yolo_classes[class_ids[i]])

            if label == 'pothole':  # Assuming 'pothole' is the class
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Pothole Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
