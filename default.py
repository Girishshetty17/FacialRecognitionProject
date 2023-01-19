import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("C:/Users/giris/OneDrive/Desktop/Dataset/yolov3.weights", "C:/Users/giris/OneDrive/Desktop"
                                                                                "/Dataset/yolov3.cfg")
classes = []
with open("C:/Users/giris/OneDrive/Desktop/Dataset/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Load image
img = cv2.imread("C:/Users/giris/OneDrive/Desktop/Dataset/defaultimage1.jpeg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
print(img.shape)
height, width, channels = img.shape

# Detect objects
blob = cv2.dnn.blobFromImage(
    img, 0.00392, (416, 16), (0, 0, 0), True, crop=False)
print(blob.shape)
net.setInput(blob)
outs = net.forward(output_layers)
#outs = [layer_names[i[0]] for i in net.getUnconnectedOutLayers()]
# Show information on the screen
class_ids = []
confidences = []
boxes = []
for out in output_layers:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw boxes
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (255, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 2)

# Show image
cv2.imshow("C:/Users/giris/OneDrive/Desktop/Dataset/defaultimage1.jpeg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
