
# import os
import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)
ClassLabels = []
file_name = 'text.txt'
with open(file_name,'rt') as fpt:
    ClassLabels = fpt.read().rstrip('\n').split('\n')
    # ClassLabels.append(fpt.read()) # another process to read the img

print(ClassLabels)

model.setInputSize (320,320)
model.setInputScale (1.0/127.5) ## 255/2 = 127.5
model.setInputMean ((127.5,127.5,127.5)) ##
model.setInputSwapRB (True)

print(len(ClassLabels))


# Read an image using OpenCV and display it in a window.

img =cv2.imread('OIP (1).jpeg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

ClassIndex, confidece, box = model.detect(img, confThreshold = 0.5)
print(ClassIndex)


# detect the object from the image

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip (ClassIndex.flatten(), confidece.flatten(), box):
    #cv2.rectangle (frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, ClassLabels [ClassInd-1], (boxes [0]+10, boxes [1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


# detect the object from the video , webcame and device camera 
# object detection from 
# video use --> cap = cv2.VideoCapture('file path')
# webcame use --> cap = cv2.VideoCapture(1)
# device camers use --> cap = cv2.VideoCapture(0)


cap = cv2.VideoCapture(0)
#Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
while True:
    ret, frame = cap.read()
    ClassIndex, confidece, bbox = model.detect (frame, confThreshold = 0.5)
    print(ClassIndex)
    if(len(ClassIndex)!= 0):
        for ClassInd, conf, boxes in zip(ClassIndex. flatten(), confidece.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, ClassLabels [ClassInd-1], (boxes [0]+10, boxes [1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness = 3)
    cv2.imshow('Object Detection Tutorial', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()