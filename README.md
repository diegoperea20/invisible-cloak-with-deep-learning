# Invisible cloak with deep learning

<p align="justify">
Invisible cloak in a traditional way with the use of color and deep learning using YOLOv8 and Maskrcnn.
</p>
<p align="center">
  <img src="README-images\traditional.PNG" alt="StepLast">
</p>
<p align="center">
  <img src="README-images\maskrcnn.PNG" alt="StepLast">
</p>


<p align="justify">
In traditional, the light blue color is detected so that the background image can be seen. 
</p>
<p align="center">
  <img src="README-images\traditional.PNG" alt="StepLast">
</p>

<p align="justify">
In deep learning in YOLOV8 an instance segmentation model is created in which light blue t-shirts are segmented, so that the mask is detected and combined with the background image.
</p>

<p align="justify">
The model was made, first using roboflow to label the dataset and export it in YOLOV8 in .zip and with that in google colab or local to perform the following commands
</p>

```python
!unzip /content/drive/MyDrive/your_folder/name_label.v1i.yolov8.zip

!pip install ultralytics

!yolo task=segment mode=train epochs=160 data=/content/drive/MyDrive/your_folder/data.yaml model=yolov8m-seg.pt imgsz=640 batch=2

```
<p align="center">
  <img src="README-images\yolo-deep.PNG" alt="StepLast">
</p>



<p align="justify">
In deep learning in Maskrcnn you use the coconut weights and mask the detected object, which in this case will be the human class '0' so that it blends with the background of the video. 
</p>

<p align="justify">
Note that you must put a video.mp4 of your current background and when you run 'mask_rcnn_segmentation.py' you must not be at the beginning of the execution but after the video background is loaded.
</p>
<p align="center">
  <img src="README-images\files-maskrcnn.PNG" alt="StepLast">
</p>
<p align="center">
  <img src="README-images\maskrcnn.PNG" alt="StepLast">
</p>

---


## Optional steps to implement

1. Use Dockerfile 
2. Use virtual enviroments and apply  requirements.txt 
```python
#virtual enviroment with conda 
conda create -n my_enviroment python=3.11.4

pip install -r requirements.txt

```