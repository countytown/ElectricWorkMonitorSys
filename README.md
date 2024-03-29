# ElectricWorkMonitorSystem
## Introduction
This is the source code of YOLOv3 version for our paper "Real-time safety distance warning framework for proximity detection based on oriented object detection and pinhole model".  More information will be available soon.
A monitor system for electrical working, including object detection, tracking, and distance estimation. As shown in pictures and the video, numbers on the green lines are distances between insulator and helmet, white nmbers are estimated distances from camera to helmet(on the projection ground). Note that the video is randomly selected so the calibration parameters are based on experience. For better performance, please try adapt the code for your own purpose in real life, and replace corresponding settings.


## Dataset
We provide the original dataset (labeled) at https://pan.baidu.com/s/1mYue8Jmjt5jzUoFmT_Sh0w, and the code is 99xn. The citation information will be available after acceptance and contact the author for usage permission before that.

## Image Demo
<!-- ![2 mp4_20220220_204000899](https://user-images.githubusercontent.com/38877851/154842848-f50b1b69-0edb-4e17-881a-ad7c370f20b7.jpg) -->
<!-- ![demo1 mp4_20220226_182504380](https://user-images.githubusercontent.com/38877851/155839938-99557328-3c45-49e5-bde9-b10c338eae11.jpg) -->

<img src="https://user-images.githubusercontent.com/38877851/154842848-f50b1b69-0edb-4e17-881a-ad7c370f20b7.jpg" width="400"/><img src="https://user-images.githubusercontent.com/38877851/155839938-99557328-3c45-49e5-bde9-b10c338eae11.jpg" width="400"/>
</center>

  
## Video Demo
https://user-images.githubusercontent.com/38877851/162617483-ce133826-9206-49dd-a3a0-179e43ace1e5.mp4  (No ground truth and no calibration is set.)
https://user-images.githubusercontent.com/38877851/200719842-22530725-3875-414f-a543-032bfab52827.mp4  (Ground truth distance is 1.2)

## Citation
Please cite the following reference for your academic research:<br>
@article{li2023fast,
  title={Fast safety distance warning framework for proximity detection based on oriented object detection and pinhole model},  
  author={Li, Hao and Qiu, Junhui and Yu, Kailong and Yan, Kai and Li, Quanjing and Yang, Yang and Chang, Rong},  
  journal={Measurement},  
  volume={209},  
  pages={112509},  
  year={2023},  
  publisher={Elsevier}  
}  

## Acknowledgement
<!-- For usage of the source code and dataset, pelase contact the author first.-->
The detection framework is based on https://github.com/ming71/rotate-yolov3.



