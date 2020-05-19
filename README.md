# Lego Detection
<br/>
<br/>
This repo is for detecting Legos on the Jetson TX2 using YOLOv3 as implemented here (https://github.com/eriklindernoren/PyTorch-YOLOv3).
<br/>
<br/>
detect.py is simply the detection algorithm, whereas findLegos.py reads in live images from a webcam and sends commands to an arduino to drive the motors towards the legos identified
<br/>
<br/>
<br/>
Create the checkpoints directory
<br/>
<code>$ mkdir checkpoints</code>
<br/>
<code>$ cd checkpoints</code>
<br/>
<br/>
Upload the following weights file to the checkpoints directory
<br/>
https://drive.google.com/file/d/10CF8gRKXVGIlZXO1pY3kiOyPIQ2iQoNZ/view?usp=sharing
<br/>
<br/>
After updating the checkpoints directory, you can run findLegos.py to locate and drive towards Legos
<br/>
Note you may be prompted to install some missing dependencies
<br/>
<br/>
<br/>
If only intending to run detection code on previously obtained images (no arduino/webcam dependency), follow the additional steps:
<br/>
<code>$ mkdir output</code>
<br/>
<code>$ cd data</code>
<br/>
<code>$ mkdir samples</code>
<br/>
<br/>
In the samples sub-directory, upload all of the images you would like to run the detection on, and then return to the root directory and run detect.py using the following command
<br/>
<code>$ python3 detect.py \--image_folder data/samples/ \--class_path data/custom/classes.names \--model_def config/yolov3-custom.cfg \--weights_path checkpoints/yolov3_ckpt_final.pth
</code>

