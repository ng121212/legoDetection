# legoDetection

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
<br/>
<br/>
If only intending to run detection code on previously obtained images (no arduino/webcam dependency), follow the additional steps:
<br/>
<code>$ mkdir output</code>
<br/>
<code>$ mkdir -p data/samples</code>
<br/>
<br/>
In the samples sub-directory, upload all of the images you would like to run the detection on, and then return to the root directory and run detect.py
