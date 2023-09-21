
from imutils.video import FPS
import numpy as np
import matplotlib.pyplot as plt
import cv2

use_gpu = 1
webcam = 1
expected_confidence = 0.3
threshold = 0.1
show_output = 1
save_output = 1
kernel = np.ones((5,5),np.uint8)


weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"


print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


if use_gpu:
   
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


print("[INFO] accessing video stream...")
if webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('video.mp4')

writer = None
fps = FPS().start()

print("[INFO] background recording...")
for _ in range(60):
    _,bg = cap.read()
print("[INFO] background recording done...")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 10,(bg.shape[1], bg.shape[0]), True)


while True:
    grabbed, frame = cap.read()
    cv2.imshow('org',frame)
    
    if not grabbed:
        break

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final","detection_masks"])

    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        if classID!=0:continue
        confidence = boxes[0, 0, i, 2]

        if confidence > expected_confidence:
            # scale the bounding box coordinates back relative to the size of the frame and then compute the width and the height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_CUBIC)
            mask = (mask > threshold)
            bwmask = np.array(mask,dtype=np.uint8) * 255
            bwmask = np.reshape(bwmask,mask.shape)
            bwmask = cv2.dilate(bwmask,kernel,iterations=2)
            # bwmask = cv2.erode(bwmask,kernel,iterations=1)
            

            
            frame[startY:endY, startX:endX][np.where(bwmask==255)] = bg[startY:endY, startX:endX][np.where(bwmask==255)]

    if show_output:
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) ==27:
            break

    if save_output:
        writer.write(frame)

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

