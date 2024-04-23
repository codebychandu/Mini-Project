import cv2

# Load pre-trained model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Read class labels
with open('coco.names', 'rt') as fpt:
    ClassLabels = fpt.read().rstrip('\n').split('\n')

# Set model input parameters
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)  # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Set confidence threshold
conf_threshold = 0.6

# OpenCV webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=conf_threshold)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(bbox, confidence, conf_threshold, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, bbox[i], (255, 0, 0), 2)
            cv2.putText(frame, ClassLabels[ClassIndex[i][0] - 1], (bbox[i][0] + 10, bbox[i][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detection overlay
    cv2.imshow('Object Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
