#!/opt/conda/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import queue

import numpy
from numpy.core.fromnumeric import compress
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
from data_to_images import draw_point
from data_to_images import draw_stickman
import cv2

import argparse
import os
import sys
import pprint
import statistics
import traceback

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from lpn.config import cfg
from lpn.config import update_config
from lpn.core.inference import get_final_preds, get_max_preds, get_final_preds_using_softargmax, get_predicitons
from lpn.utils.utils import create_logger, get_model_summary
from lpn.models.lpn import get_pose_net

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

import lpn.dataset
import lpn.models

import threading


class HumanPoseEstimationROS():

    def __init__(self, frequency, args):

        rospy.init_node("hpe_lpn", log_level=rospy.DEBUG)
        
        self.rate = rospy.Rate(int(frequency))

        # Update configuration file
        reset_config(cfg, args)
        print("args are: {}".format(args))
        update_config(cfg, args)
        # Add config to class variable to enable propagation to run method
        self.cfg = cfg

        # Legacy CUDNN --> probably not necessary 
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        self.model_ready = False
        self.first_img_reciv = False
        self.nn_input_formed = False
        self.nn_output_formed = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.device = torch.device("cpu")

        rospy.loginfo("[HPE-LPN] Loading model")
        self.model = self._load_model(cfg)
        rospy.loginfo("[HPE-LPN] Loaded model...")
        self.model_ready = True

        # If use depth (use Xtion camera) 
        # self.use_depth = eval(str(args.use_depth))
        self.use_depth = eval(str(args.use_depth))
        self.use_depth = False 
        rospy.loginfo("[HPE-LPN] Using depth camera: {}".format(self.use_depth))
        
        # Initialize subscribers/publishers
        self._init_publishers()
        self._init_subscribers()

        self.nn_input = None
        
        # Initialize variables for darknet person bounding boxes
        self.x = None
        self.y = None
        self.w = None
        self.h = None

        # Initialize font
        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/include/arial.ttf", 20, encoding="unic")

        self.filtering_active = False; self.filter_first_pass = False

        # If HMI integration (use compressed image)
        self.compressed_stickman = False


        # NN transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            normalize])


    def _init_subscribers(self):
        
        if self.use_depth:
            # Xtion Cam
            self.camera_sub = rospy.Subscriber("camera/rgb/image_raw", Image, self.image_cb, queue_size=1)
        else:
            # USB Cam
            # TODO: Send as param in launch file 
            self.camera_sub = rospy.Subscriber("usb_camera/image_raw", Image, self.image_cb, queue_size=1)
            
        #self.darknet_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_cb, queue_size=1)

    def _init_publishers(self):
        self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        self.image_compressed_pub = rospy.Publisher("/stickman_compressed", CompressedImage, queue_size=1)
        self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)

    def _load_model(self, config):
        
        rospy.loginfo("Model name is: {}".format(config.MODEL.NAME))
        # Eval method is used to load different models! 
        model = get_pose_net(config, is_train=False)

        rospy.loginfo("Passed config is: {}".format(config))

        # Hacked! -> incorrect model file name 
        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
            rospy.loginfo('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
            rospy.loginfo('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        model.eval()
        model.to(self.device)           
        
        return model

    def image_cb(self, msg):

        start_time = rospy.Time.now().to_sec()

        self.first_img_reciv = True

        debug_img = False
        if debug_img:
            rospy.loginfo("Image width: {}".format(msg.width))
            rospy.loginfo("Image height: {}".format(msg.height))
            rospy.loginfo("Data is: {}".format(len(msg.data)))
            rospy.loginfo("Input shape is: {}".format(input.shape))

        # Transform img to numpy array        
        start_time = rospy.Time.now().to_sec()
        self.org_img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)
        self.img_w = msg.width; self.img_h = msg.height

        self.cam_img = cv2.resize(self.org_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        self.center = 256/2, 256/2
        # Check this scaling
        self.scale = numpy.array([1, 1], dtype=numpy.float32) 

        # Transform img to 
        self.nn_input = self.transform(self.cam_img).unsqueeze(0).to(self.device)   

        #rospy.logdebug("Duration of conversion before inference is: {}".format(rospy.Time().now().to_sec() - start_time))        
        # self.nn_input_formed = True
        # Create whole pipeline in image callback to reduce neccessity for RUN method
        # output = self.model(self.nn_input)
        # self.preds = get_predicitons(output.detach().cpu().numpy(), scaling=True)
        self.nn_output_formed = True
        
        if debug_img:
            rospy.loginfo("NN_INPUT {}".format(self.nn_input))             

        duration = rospy.Time.now().to_sec() - start_time 
        rospy.logdebug("Duration {} of image_cb is: {}".format(threading.current_thread(), duration)) # max --> 0.01s                   
  
    def filter_predictions(self, predictions, filter_type="avg", w_size=3): 

        # Different indexing between SB and LPN
        preds_ = predictions

        for i, prediction in enumerate(preds_): 
            # Keypoint 10 should be left wrist
            if i == 10: 
                l_x = preds_[i][0]; l_y = preds_[i][1]; 

            # Keypoint 15 should be right wrist
            if i == 15:
                r_x = preds_[i][0]; r_y = preds_[i][1]; 

        
        p_right = (r_x, r_y); p_left = (l_x, l_y)
        filtered_lx, filtered_ly, filtered_rx, filtered_ry = self.filtering(p_right, p_left, type=filter_type, window_size=w_size)

        preds_[10][0] = int(filtered_lx); preds_[10][1] = int(filtered_ly); 
        preds_[15][0] = int(filtered_rx); preds_[15][1] = int(filtered_ry); 

        return preds_

    def filtering(self, p_right, p_left, type="avg", window_size=3):

        if self.first_img_reciv and not self.filter_first_pass : 
            self.left_x = []; self.left_y = []; 
            self.right_x = []; self.right_y = [];          
            self.filter_first_pass = True 
        
        else:
            self.left_x.append(p_left[0]); self.left_y.append(p_left[1])
            self.right_x.append(p_right[0]); self.right_y.append(p_right[1])

            if len(self.left_x) >= window_size: 
                self.filtering_active = True
                # Cropped
                crop_l_x = self.left_x[-1 - window_size:].copy(); crop_l_y = self.left_y[-1 - window_size:].copy()
                crop_r_x = self.right_x[-1 - window_size:].copy(); crop_r_y = self.right_y[-1 - window_size:].copy()

                if type == "avg": 

                    filter_l_x = HumanPoseEstimationROS.avg_list(crop_l_x); filter_l_y = self.avg_list(crop_l_y)
                    filter_r_x = HumanPoseEstimationROS.avg_list(crop_r_x); filter_r_y = self.avg_list(crop_r_y)

                elif type == "median":

                    filter_l_x = HumanPoseEstimationROS.median_list(crop_l_x); filter_l_y = self.median_list(crop_l_y)
                    filter_r_x = HumanPoseEstimationROS.median_list(crop_r_x); filter_r_y = self.median_list(crop_r_y)

                buffer = 10
                if len(self.left_x) > buffer:
                    self.left_x = self.left_x[-buffer:]; self.left_y = self.left_y[-buffer:]
                    self.right_x = self.right_x[-buffer:]; self.right_y = self.right_y[-buffer:]              

        if self.filtering_active:

            return filter_l_x, filter_l_y, filter_r_x, filter_r_y

        else: 

            return p_left[0], p_left[1], p_right[0], p_right[1]       
            
    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        self.first = True

        while not self.model_ready:
            self.logger.info("LPN model for HPE not ready.")
        
        while not rospy.is_shutdown(): 
            
            if (self.first_img_reciv and self.nn_output_formed):                

                start_time = rospy.Time.now().to_sec()
                
                # Convert ROS Image to PIL 
                start_time1 = rospy.Time.now().to_sec()
                pil_img = PILImage.fromarray(self.org_img.astype('uint8'), 'RGB')
                rospy.logdebug("Conversion to PIL Image from numpy: {}".format(rospy.Time.now().to_sec() - start_time1))
                
                # Get NN Output ## TODO: Check if this could be made shorter :) 
                start_time2 = rospy.Time.now().to_sec()
                output = self.model(self.nn_input) 
                #rospy.logdebug("NN inference1 duration: {}".format(rospy.Time.now().to_sec() - start_time2))

                # Heatmaps
                # torch.cuda.synchronize()
                start_time3 = rospy.Time.now().to_sec()
                # Detach basically detaches computational graph which contains gradient descent info
                batch_heatmaps = output.detach().cpu().numpy()
                #rospy.logdebug("Duration of cpu().detach().numpy() is: {}".format(rospy.Time.now().to_sec() - start_time3))

                #HPERos.save_heatmaps(batch_heatmaps)
                # preds, maxvals = get_max_preds(batch_heatmaps)
                self.preds = get_predicitons(batch_heatmaps, scaling=True)
                # Doesn't look faster
                #self.scale = [self.img_w/64, self.img_h/64]
                #preds = get_final_preds_using_softargmax(self.cfg, output, self.center, self.scale)
                
                debug_predictions = False
                if debug_predictions:
                    rospy.logdebug(str(preds[0][0][0]) + "  " + str(preds[0][0][1]))                
                    rospy.logdebug("Preds are: {}".format(preds))     
                    rospy.logdebug("Preds shape is: {}".format(preds.shape))
                
                # Filtering s
                try:
                    preds = self.filter_predictions(self.preds, "avg", 3)
                except Exception as e:
                    print(e)

                # Draw stickman
                start_time5 = rospy.Time.now().to_sec()
                stickman = HumanPoseEstimationROS.draw_stickman(pil_img, preds)
                
                # If compressed_stickman (zones don't work, no subscriber on compressed)
                if self.compressed_stickman: 
                    stickman_compressed_msg = HumanPoseEstimationROS.convert_pil_to_ros_compressed(stickman)
                    self.image_compressed_pub.publish(stickman_compressed_msg)
                else:
                    stickman_ros_msg = HumanPoseEstimationROS.convert_pil_to_ros_img(stickman)
                    self.image_pub.publish(stickman_ros_msg)
                    

                # Prepare predictions for publishing - convert to 1D float list
                converted_preds = []
                for pred in preds:
                    converted_preds.append(pred[0])
                    converted_preds.append(pred[1])
                preds_ros_msg = Float64MultiArray()
                preds_ros_msg.data = converted_preds
                
                self.pred_pub.publish(preds_ros_msg)

                duration = rospy.Time.now().to_sec() - start_time
                debug_runtime = False
                if debug_runtime:
                    rospy.loginfo("Run {} duration is: {}".format(threading.current_thread(), duration))
                
            
            #self.rate.sleep()

    @staticmethod
    def save_heatmaps(heatmaps): 
        
        num_heatmaps = heatmaps.shape[1]
        for i in range(num_heatmaps):
            numpy.savetxt('/home/developer/catkin_ws/src/hpe_ros_package/heatmaps/hm_{}.csv'.format(i), heatmaps[0, i, :, :], delimiter=',')

    @staticmethod        
    def avg_list(list_data):

        return sum(list_data)/len(list_data)

    @staticmethod
    def median_list(list_data):

        return statistics.median(list_data)

    @staticmethod
    def draw_stickman(img, predictions):
        

        draw  = ImageDraw.Draw(img)

        font_ = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/include/arial.ttf", 20, encoding="unic")

        point_r = 4

        ctl_indices = [10, 15]
        for i in range (0, len(predictions)): 
            
            if i in ctl_indices:

                if i == 10 or i == 15: 
                    fill_ = (0, 255, 0)
                    point_r = 6

            else:
                fill_ = (153, 255, 255)
                point_r = 4
            draw.ellipse([(predictions[i][0] - point_r, predictions[i][1] - point_r), (predictions[i][0] + point_r, predictions[i][1] + point_r)], fill=fill_, width=2*point_r)

            if i < len(predictions) - 1 and i != 5 and i != 9:      
                draw.line([(predictions[i][0], predictions[i][1]), (predictions[i + 1][0], predictions[i + 1][1])], fill=(153, 255, 255), width=2)


        return img

    #TODO: add those methods to img_utils --> when I have all neccessary methods for img_utils
    @staticmethod
    def convert_pil_to_ros_img(img):
        """Function for converting pillow to ros image
        Args:
            img (PIL.Image.Image): Pillow image that represents GUI
        Returns:
            sensor_msgs.msg._Image.Image: ROS image for image publisher
        """
        img = img.convert('RGB')
        msg = Image()
        stamp = rospy.Time.now()
        msg.height = img.height
        msg.width = img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img.width
        msg.data = numpy.array(img).tobytes()
        return msg   

    @staticmethod
    def convert_pil_to_ros_compressed(img, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 
        rgb_img = np_img[:, :, [2, 1, 0]]
        
        compressed_img = cv2.imencode(".{}".format(compression_type), rgb_img)[1]
        msg.data = compressed_img.tobytes()

        return msg

# Method with global params
def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.modelFile: 
        config.MODEL_FILE = args.modelFile

# Dirty hack for correct parsing of boolean arguments wth argparse
#def str2bool(v):
#  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        # default="experiments/mpii/lpn/lpn50_256x256_gd256x2_gc.yaml")
                        default="experiments/mpii/lpn/lpn50_256x192_gd256x2_gc.yaml")
    # Needed args!
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--modelFile', 
                        help="model file", 
                        type=str, 
                        default='')
    # training
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    # inference cam
    parser.add_argument('--use-depth', 
                        help="use depth cam", 
                        default=False, 
                        type=bool)

    args, unknown = parser.parse_known_args(sys.argv[2:])
 

    HPERos = HumanPoseEstimationROS(20, args)
    HPERos.run()
