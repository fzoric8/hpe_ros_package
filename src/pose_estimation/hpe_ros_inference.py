#!/opt/conda/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from numpy.core.fromnumeric import compress
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CompressedImage
from data_to_images import draw_point
from data_to_images import draw_stickman
import cv2

import argparse
import sys
import pprint
import statistics

# Explicitly set number of threads used by this script
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=1

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds, get_max_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

import dataset
import models
import io


# Tracing execution:
# https://github.com/boschresearch/ros1_tracetools 

from std_msgs.msg import Bool

class HumanPoseEstimationROS():

    def __init__(self, frequency, args):


        rospy.init_node("hpe_simplebaselines")
        
        self.rate = rospy.Rate(int(frequency))

        # Update configuration file
        update_config(args.cfg)
        reset_config(config, args)
        self.config = config

        # Legacy CUDNN --> probably not necessary 
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        self.model_ready = False
        self.first_img_reciv = False
        self.nn_input_formed = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        rospy.loginfo("[HPE-SimpleBaselines] Loading model")
        self.model = self._load_model(config)
        rospy.loginfo("[HPE-SimpleBaselines] Loaded model...")
        self.model_ready = True

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

        # Image compression
        self.use_compressed_img = False
        self.use_normal_img = not self.use_compressed_img

        # Initialize subscribers/publishers 
        self._init_publishers(); self._init_subscribers()

    def _init_subscribers(self):

        if self.use_compressed_img:
            rospy.loginfo("Subscribing to compressed image!")
            self.camera_sub = rospy.Subscriber("usb_camera/image_raw/compressed", CompressedImage, self.compressed_img_cb, queue_size=1)
        
        if self.use_normal_img:
            rospy.loginfo("Subscribing to normal image!")
            self.camera_sub = rospy.Subscriber("usb_camera/image_raw", Image, self.image_cb, queue_size=1)
        #self.darknet_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_cb, queue_size=1)

    def _init_publishers(self):
        self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        self.image_compressed_pub = rospy.Publisher("/stickman_compressed", CompressedImage, queue_size=1)
        self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)
        self.img_capt_pub = rospy.Publisher("/test_pub", Image, queue_size=1)

    def _load_model(self, config):
        
        rospy.loginfo("Model name is: {}".format(config.MODEL.NAME))
        model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False)

        rospy.loginfo("Passed config is: {}".format(config))
        rospy.loginfo("config.TEST.MODEL.FILE")

        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
            rospy.loginfo('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
            rospy.loginfot('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        model.to(self.device)           
        
        return model

    def image_cb(self, msg):

        #rospy.sleep(0.1)

        # Get travel time (time in which image has been captured to recieving in callback)
        rospy.loginfo("Travel time is: {}".format(self.get_travel_time(msg.header)))
        self.img_capt_pub.publish(msg)

        start_time = rospy.Time.now().to_sec()

        debug_img = False
        if debug_img:
            rospy.loginfo("Image width: {}".format(msg.width))
            rospy.loginfo("Image height: {}".format(msg.height))
            rospy.loginfo("Data is: {}".format(len(msg.data)))
            rospy.loginfo("Input shape is: {}".format(input.shape))

        # Transform img to numpy array        
        self.org_img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)

        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Tensor transformations
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        #if self.x != None and self.y != None:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, self.x, self.y, self.w, self.h)
        #else:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, 0, 0, msg.width, msg.height)

        self.cam_img = cv2.resize(self.org_img, dsize=(348,348), interpolation=cv2.INTER_CUBIC)
        
        #self.logger.info("Scale is: {}".format(self.scale))
        self.center = 384/2, 384/2

        # https://github.com/microsoft/human-pose-estimation.pytorch/issues/26
        
        # Check this scaling
        self.scale = numpy.array([1, 1], dtype=numpy.float32) 
        
        # Transform img to 
        self.nn_input = transform(self.cam_img).unsqueeze(0).to(self.device)   

        self.first_img_reciv = True; self.nn_input_formed = True
        
        if debug_img:
            rospy.loginfo("NN_INPUT {}".format(self.nn_input))             

        duration = rospy.Time.now().to_sec() - start_time 
        
        rospy.loginfo("Duration of image_cb is: {}".format(duration)) # max --> 0.01s

    def compressed_img_cb(self, msg): 

        rospy.sleep(0.1)

        rospy.loginfo("Compressed image travel time is: {}".format(self.get_travel_time(msg.header)))

        start_time = rospy.Time.now().to_sec()


        self.org_img = numpy.array(PILImage.open(io.BytesIO(bytearray(msg.data))), dtype=numpy.uint8)

        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Tensor transformations
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        #if self.x != None and self.y != None:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, self.x, self.y, self.w, self.h)
        #else:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, 0, 0, msg.width, msg.height)

        self.cam_img = cv2.resize(self.org_img, dsize=(348,348), interpolation=cv2.INTER_CUBIC)
        
        #self.logger.info("Scale is: {}".format(self.scale))
        self.center = 384/2, 384/2

        # https://github.com/microsoft/human-pose-estimation.pytorch/issues/26
        
        # Check this scaling
        self.scale = numpy.array([1, 1], dtype=numpy.float32) 
        
        # Transform img to 
        self.nn_input = transform(self.cam_img).unsqueeze(0).to(self.device)   

        self.first_img_reciv = True; self.nn_input_formed = True
             

        duration = rospy.Time.now().to_sec() - start_time 

        rospy.loginfo("Duration of compressed image_cb is: {}".format(duration)) # max --> 0.01s
                         
    def darknet_cb(self, darknet_boxes):
        
        max_area = 0
        for bbox in darknet_boxes.bounding_boxes: 
            if bbox.Class == "person":
                this_area = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
                if this_area > max_area:
                    max_area = this_area
                    self.x = bbox.xmin
                    self.y = bbox.ymin
                    self.w = bbox.xmax - bbox.xmin
                    self.h = bbox.ymax - bbox.ymin
        else:
            return

    def aspect_ratio_scaler(self, img, x0, y0, width, height):

        box = [x0, y0, width, height]
        x,y,w,h = box[:4]
        
        center = numpy.zeros((2), dtype=numpy.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = width * 1.0 / height
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = numpy.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=numpy.float32)
        if center[0] != -1:
            scale = scale * 1.25

        r = 0
        trans = get_affine_transform(center, scale, r, config.MODEL.IMAGE_SIZE)
        scaled_img = cv2.warpAffine(
            img,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        #scaled_img = numpy.array(scaled_img, dtype=numpy.uint8)

        return scaled_img, center, scale

    def filter_predictions(self, predictions, filter_type="avg", w_size=3): 


        preds_ = predictions[0]

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
            
    def get_travel_time(self, msg_header): 

        capture_time = msg_header.stamp.secs

        travel_time = rospy.Time().now().to_sec() - capture_time

        return travel_time
    
    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        self.first = True

        while not self.model_ready:
            self.logger.info("SimpleBaselines model for HPE not ready.")
        
        while not rospy.is_shutdown(): 

            if (self.first_img_reciv and self.nn_input_formed):
               
                start_time = rospy.Time.now().to_sec()
                
                # Convert ROS Image to PIL 
                start_time1 = rospy.Time.now().to_sec()
                pil_img = PILImage.fromarray(self.org_img.astype('uint8'), 'RGB')
                rospy.logdebug("Conversion to PIL Image from numpy: {}".format(rospy.Time.now().to_sec() - start_time1))
                
                # Get NN Output ## TODO: Check if this could be made shorter :) 
                rospy.logdebug(type(self.nn_input))
                start_time2 = rospy.Time.now().to_sec()
                output = self.model(self.nn_input) 
                rospy.logdebug("NN inference1 duration: {}".format(rospy.Time.now().to_sec() - start_time2))

                # Heatmaps
                start_time3 = rospy.Time.now().to_sec()
                batch_heatmaps = output.cpu().detach().numpy()
                # Get predictions                
                #preds, maxvals = get_final_preds(config, batch_heatmaps, self.center, self.scale)
                # preds otuput is list of lists, list that contain list that contains 16 points!
                preds, maxvals = get_max_preds(batch_heatmaps)
                rospy.logdebug("NN inference2 duration: {}".format(rospy.Time.now().to_sec() - start_time3))

                # Heatmap size is 88x88, so this scales predictions to image size 
                for pred in preds[0]:
                    pred[0] = pred[0]  * (640/88)
                    pred[1] = pred[1]  * (480/88)
                rospy.logdebug(str(preds[0][0][0]) + "   " + str(preds[0][0][1]))
                
                rospy.logdebug("Preds are: {}".format(preds))     
                rospy.logdebug("Preds shape is: {}".format(preds.shape))
                # Preds shape is [1, 16, 2] (or num persons is first dim)
                # rospy.loginfo("Preds shape is: {}".format(preds[0].shape))

                preds = self.filter_predictions(preds, "avg", 5)

                # Draw stickman
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
                debug_runtime = True
                if debug_runtime:
                    rospy.loginfo("Run duration is: {}".format(duration))

            
            self.rate.sleep()

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
                    fill_ = "green"

            else:
                fill_ = (153, 255, 255)
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

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--shift-heatmap', 
                        help="shift heatmap", 
                        default=False, 
                        type=bool)

    args, unknown = parser.parse_known_args(sys.argv[2:])
 

    HPERos = HumanPoseEstimationROS(20, args)
    HPERos.run()
