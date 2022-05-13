# hpe_ros_package


ROS package for human pose estimation with [Microsoft SimpleBaselines](https://github.com/microsoft/human-pose-estimation.pytorch) algorithm.

Input to the network should be cropped image of a person (detected person). 


You need to download weights for COCO or MPII dataset from following [link](https://onedrive.live.com/?authkey=%21AKqtqKs162Z5W7g&id=56B9F9C97F261712%2110692&cid=56B9F9C97F261712). 

You need [usb-cam](https://github.com/ros-drivers/usb_cam) ROS package for testing it on your PC with webcam. 

### Starting procedure

Launch your cam, in this case web-cam: 
```
roslaunch usb_cam usb_cam-test.launch 
``` 
After downloading weights to `/home/developer/catkin_ws/src/hpe_ros_package/src/lib/models` you can run your HPE inference 
with following: 
```
roslaunch hpe webcam_inference.launch
```

### HMI starting procedure 

Starting procedure for HMI is to setup `ROS_MASTER_URI` and `ROS_IP` to the values of host PC to enable 
streaming to AR glasses. 
After that we can run following script: 

```
roslaunch hpe hmi_integration.launch 
```

It's not neccessary to setup ROS_MASTER_URI script anymore because currently stream is being sent using RTSP protocol. 


### Transport Server -> Client 

Current transport from server to client currently takes:

If we do not compress image. 

```
average rate: 9.519
	min: 0.041s max: 0.244s std dev: 0.02216s window: 2050

```

While on raspberry we have subscription frequency `2.5 Hz`. 

After using image compression with `image_transport` I get same values. 

Launch for using compressed is: 
```
 rosrun image_transport republish raw in:=stickman_cont_area out:=stickman_compressed
```

### Current implementation status 

Currently it's possible to use human pose estimation for generating references for UAV. 

Now we have attitude zone control which sends commands to UAV in joy msg type. 
It's possible to easily implement position control also. 

### Neural networks 

Currently there are two implemented neural networks for human pose estimation. 
One is SimpleBaselines and another one is LPN which is somewhat lightweight 
version of SimpleBaselines. 

#### Open challenges

1. How to run inference in realtime? (Threading issue) 
2. How to retrain network to have fewer GFLOPs? 
3. How to solve stereovision problem with this? 

#### TODO: 

Initial ideas that could result in faster inference are: 

1. Train new neural network that can take in consideration both images 
2. Setup MLFlow on Server PC
3. Test newly trained neural network


#### LPN and SimpleBaselines comparison 

| SimpleBaselines | Params | FLOPS |
| ----------- | -----------|------------|
| ResNet50    | 34M        | 8.96       |
| ResNet101   | 53M        | 12.46      |
| ResNet152   | 68.6M      | 15.76      |
| LPN |Params | FLOPS | 
|ResNet50 |  2.9M  | 1.06 |
|ResNet101 | 5.3M  | 1.46  |
|ResNet152 | 7.4M  | 1.86  |

### Neural networks for HPE

Available neural networks for HPE which are similar to SimpleBaselines are: 
 - [HRNet](https://github.com/HRNet) 
 - [LPN-Simple and lightweight pose estimation](https://github.com/zhang943/lpn-pytorch) 

My GPU is currently [NVIDIA GeForce MX330](https://www.techpowerup.com/gpu-specs/geforce-mx330.c3493) 
which has 1.224 GFLOPS by specification. 

### Current status

On my PC with NVIDIA GeForce MX330, achieved inference time with LPN network is around 5Hz. 

### TODO High priority: 

 - [ ] Check launch files / Add depth and control types arguments 
 - [ ] Add calibration and dynamic control to arguments in launch files
 - [ ] Finish depth control (pitch/depth control)
 - [ ] Try HRNet --> Too slow for our use-case 

### TODO Low priority: 

 - [ ] Try 3d pose estimation 
 - [ ] Add simple object detection (`darknet_ros`) for better hpe  
 - [ ] Implement correct post processing as stated [here](https://github.com/microsoft/human-pose-estimation.pytorch/issues/26) 
 
