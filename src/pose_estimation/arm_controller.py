#!/opt/conda/bin/python3
from audioop import avg
from multiprocessing import reduction
import queue
from turtle import width
import rospy
import rospkg
import sys
import cv2
import numpy 
import copy

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32, Float32, Bool
from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from img_utils import *

# TODO:
# - think of behavior when arm goes out of range! --> best to do nothing, just send references when there's signal from HPE 
# - add calibration method
# - decouple depth and zone calibration 
# - test 2D zones

class uavController:

    def __init__(self, frequency, control_type):

        nn_init_time_sec = 10
        rospy.init_node("arm_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        # Available control types are: euler, euler2d 
        if control_type: 
            self.control_type =  control_type
        else: 
            self.control_type = "euler2d"
        # Copy control type    
        self.init_control_type = copy.deepcopy(self.control_type)

        self._init_publishers()
        self._init_subscribers()

        # Define zones / dependent on input video image (Can be extracted from camera_info) 
        self.height = 480; 
        self.width = 640; 

        # Decoupled control  
        if self.control_type == "position": 
            
            # 1D control zones
            self.ctl_zone = self.define_ctl_zone((self.width/2, self.height/2), 0.5, 0.5)
            self.deadzone = self.define_deadzone(self.ctl_zone, 0.25)

        self.start_position_ctl = False

        self.rate = rospy.Rate(int(frequency))     

        # Debugging arguments
        self.inspect_keypoints = False
        self.recv_pose_meas = False        

        # Image compression for human-machine interface
        self.hmi_compression = False
        # If calibration determine zone-centers
        self.start_calib = False
        # If use depth
        self.use_depth = False
        self.depth_recv = False
        self.depth_pcl_recv = False

        # Initialize start calib time to very large value to start calibration when i publish to topic
        self.calib_duration = 10
        self.rhand_calib_px, self.rhand_calib_py = [], []
        self.lhand_calib_px, self.lhand_calib_py = [], []
        self.rshoulder_px, self.rshoulder_py = [], []
        self.lshoulder_px, self.lshoulder_py = [], []
        self.calib_depth = []
        
        # Flags for run method
        self.initialized = True
        self.prediction_started = False

        rospy.loginfo("Initialized!")   

    def _init_publishers(self): 
        
        
        #TODO: Add topics to yaml file
        if self.control_type == "position":
            self.pose_pub = rospy.Publisher("bebop/pos_ref", Pose, queue_size=1)

        if self.control_type == "euler" or self.control_type == "euler2d": 
            self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=1)

        self.stickman_area_pub = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)
        self.stickman_compressed_area_pub = rospy.Publisher("/stickman_compressed_ctl_area", CompressedImage, queue_size=1)

        # Points
        self.lhand_x_pub = rospy.Publisher("hpe/lhand_x", Int32, queue_size=1)
        self.rhand_x_pub = rospy.Publisher("hpe/rhand_x", Int32, queue_size=1)
        self.lhand_y_pub = rospy.Publisher("hpe/lhand_y", Int32, queue_size=1)
        self.rhand_y_pub = rospy.Publisher("hpe/rhand_y", Int32, queue_size=1)
        # Depths
        self.d_wrist_pub = rospy.Publisher("hpe/d_wrist", Float32, queue_size=1)
        self.d_shoulder_pub = rospy.Publisher("hpe/d_shoulder", Float32, queue_size=1)
        self.d_relative_pub = rospy.Publisher("hpe/d_relative", Float32, queue_size=1)

    def _init_subscribers(self): 

        self.preds_sub          = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        self.current_pose_sub   = rospy.Subscriber("uav/pose", PoseStamped, self.curr_pose_cb, queue_size=1)
        self.start_calib_sub    = rospy.Subscriber("start_calibration", Bool, self.calib_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth/image", Image, self.depth_cb, queue_size=1)
        self.depth_pcl_sub      = rospy.Subscriber("camera/depth/points", PointCloud2, self.depth_pcl_cb, queue_size=1)

        # stickman 
        self.stickman_sub       = rospy.Subscriber("stickman", Image, self.draw_zones_cb, queue_size=1)
           
    def publish_predicted_keypoints(self, rhand, lhand): 

        rhand_x, rhand_y = rhand[0], rhand[1]; 
        lhand_x, lhand_y = lhand[0], lhand[1]

        rospy.logdebug("rhand: \t x: {}\t y: {}".format(rhand_x, rhand_y))
        rospy.logdebug("lhand: \t x: {}\t y: {}".format(lhand_x, lhand_y))

        self.lhand_x_pub.publish(int(lhand_x))
        self.lhand_y_pub.publish(int(lhand_y))
        self.rhand_x_pub.publish(int(rhand_x))
        self.rhand_y_pub.publish(int(rhand_y))

    def average_depth_cluster(self, px, py, k, config="WH"): 

        indices = []
        start_px = int(px - k); stop_px = int(px + k); 
        start_py = int(py - k); stop_py = int(py + k); 

        # Paired indices
        for px in range(start_px, stop_px, 1): 
                for py in range(start_py, stop_py, 1): 
                    # Row major indexing
                    if config == "WH": 
                        indices.append((px, py))
                    # Column major indexing
                    if config == "HW":
                        indices.append((py, px))
            
        # Fastest method for fetching specific indices!
        depths = pc2.read_points(self.depth_pcl_msg, ['z'], False, uvs=indices)
        
        try:

            depths = numpy.array(list(depths), dtype=numpy.float32)
            depth_no_nans = list(depths[~numpy.isnan(depths)])

            if len(depth_no_nans) > 0:                
                
                avg_depth = sum(depth_no_nans) / len(depth_no_nans)
                rospy.logdebug("{} Average depth is: {}".format(config, avg_depth))
                return avg_depth

            else: 

                return None
        
        except Exception as e:
            rospy.logwarn("Exception occured: {}".format(str(e))) 
            
            return None

    def define_ctl_zone(self, center, w, h): 

        if abs(w) > 1 or abs(h) > 1: 
            rospy.logerr("Please specify width and height of control zone in percentage between 0-1")

        left_x, left_y = center[0] - (w/2) * self.width, center[1] - (h/2) * self.height
        right_x, right_y = center[0] + (w/2) * self.width, center[1] + (h/2) * self.height
        
        return ((left_x, left_y), (right_x, right_y))
    
    def define_deadzone(self, ctl_zone, percentage):

        w = ctl_zone[1][0] + ctl_zone[0][0]
        h = ctl_zone[1][1] + ctl_zone[0][1]

        w_ = ctl_zone[1][0] - ctl_zone[0][0]
        h_ = ctl_zone[1][1] - ctl_zone[0][1]

        center_x = w/2
        center_y = h/2

        left_x, left_y = center_x - (percentage * w)/2, center_y - (percentage * h)/2
        right_x, right_y = center_x + (percentage * w)/2, center_y + (percentage * h)/2
                

        return ((left_x, left_y), (right_x, right_y))
        

    # 1D checking if in range  
    def check_if_in_range(self, value, min_value, max_value): 

        if (value >= min_value and value <= max_value): 
            return True

        else: 
            return False 
    
    # 2D checking inf in range
    def in_zone(self, point, rect): 

        x, y   = point[0], point[1]
        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]

        x_cond = True if (x >= x0 and x <= x1) else False
        y_cond = True if (y >= y0 and y <= y1) else False

        if x_cond and y_cond:
            return True
        else:
            return False

    def in_ctl2d_zone(self, point, rect, deadzone): 

        x, y = point[0], point[1]
        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]
        cx, cy = (x1 + x0) / 2, (y1 + y0) / 2
        
        if self.in_zone(point, rect): 

            if abs(cx - x) > (deadzone[1][0] - cx): 
                norm_x_diff = (x - cx) / ((x1 - x0) / 2)
            else: 
                norm_x_diff = 0.0

            if abs(cy - y) > (deadzone[1][1] - cy): 
                norm_y_diff = (y - cy) / ((y1 - y0) / 2)
            else: 
                norm_y_diff = 0.0
        
        else: 

            norm_x_diff, norm_y_diff = 0.0, 0.0

        return norm_x_diff, norm_y_diff


    def zones_calibration(self, right, left, done):
        
        if not done: 
            self.rhand_calib_px.append(right[0]), self.rhand_calib_py.append(right[1])
            self.lhand_calib_px.append(left[0]), self.lhand_calib_py.append(left[1])
        
        else: 

            avg_rhand = (int(sum(self.rhand_calib_px)/len(self.rhand_calib_px)), int(sum(self.rhand_calib_py)/len(self.rhand_calib_py)))
            avg_lhand = (int(sum(self.lhand_calib_py)/len(self.lhand_calib_px)), int(sum(self.lhand_calib_py)/len(self.lhand_calib_py)))

            return avg_rhand, avg_lhand

    def average_zone_points(self, rshoulder, lshoulder, avg_len): 

        self.rshoulder_px.append(rshoulder[0]); self.rshoulder_py.append(rshoulder[1])
        self.lshoulder_px.append(lshoulder[0]); self.lshoulder_py.append(lshoulder[1])

        if len(self.rshoulder_px) > avg_len: 
            avg_rshoulder_px = int(sum(self.rshoulder_px[-avg_len:])/len(self.rshoulder_px[-avg_len:]))
            avg_rshoulder_py = int(sum(self.rshoulder_py[-avg_len:])/len(self.rshoulder_py[-avg_len:]))
            avg_lshoulder_px = int(sum(self.lshoulder_px[-avg_len:])/len(self.lshoulder_px[-avg_len:]))
            avg_lshoulder_py = int(sum(self.lshoulder_py[-avg_len:])/len(self.lshoulder_py[-avg_len:]))
        else: 
            avg_rshoulder_px = int(sum(self.rshoulder_px)/len(self.rshoulder_px))
            avg_rshoulder_py = int(sum(self.rshoulder_py)/len(self.rshoulder_py))
            avg_lshoulder_px = int(sum(self.lshoulder_px)/len(self.lshoulder_px))
            avg_lshoulder_py = int(sum(self.lshoulder_py)/len(self.lshoulder_py))

        return ((avg_rshoulder_px, avg_rshoulder_py), (avg_lshoulder_px, avg_lshoulder_py))

    def depth_minmax_calib(self, collected_data): 
         
        min_data = min(collected_data)
        max_data = max(collected_data)
        data_range = max_data - min_data

        return min_data, max_data, data_range

    def depth_avg_calib(self, collected_data): 
                     
        avg = sum(collected_data)/len(collected_data)

        return avg

    def depth_data_collection(self, px, py, done=False):

        if not done:
            depth = self.average_depth_cluster(px, py, 2, "WH")
            if depth: 
                self.calib_depth.append(depth)
        else: 

            #avg_depth  = sum(self.calib_depth) / len(self.calib_depth)
            
            # Return collected data
            return self.calib_depth


    def run(self): 

        while not rospy.is_shutdown():
            if not self.initialized or not self.prediction_started: 
                rospy.logdebug("Waiting prediction")
                rospy.sleep(0.1)
            else:

                # Reverse mirroring operation: 
                lhand_ = (abs(self.lhand[0] - self.width), self.lhand[1])
                rhand_ = (abs(self.rhand[0] - self.width), self.rhand[1])

                # ========================================================
                # ===================== Calibration ======================
                if self.start_calib:

                    pass
                    # TODO: Implement calibration here

                if self.control_type == "position": 
                    
                    rospy.logdebug("rhand_: {}".format(rhand_))
                    rospy.logdebug("ctl_rect_: {}".format(self.ctl_zone))
                    rospy.logdebug("deadzone: {}".format(self.deadzone))

                    x_cmd, y_cmd = self.in_ctl2d_zone(rhand_, self.ctl_zone, self.deadzone)
                    rospy.logdebug("x_cmd: {}".format(x_cmd))
                    rospy.logdebug("y_cmd: {}".format(y_cmd))
            

                self.rate.sleep()


    def curr_pose_cb(self, msg):
        
        self.recv_pose_meas = True; 
        self.current_pose = PoseStamped(); 
        self.current_pose.header = msg.header
        self.current_pose.pose.position = msg.pose.position
        self.current_pose.pose.orientation = msg.pose.orientation

    def pred_cb(self, converted_preds):
        preds = []

        start_time = rospy.Time().now().to_sec()
        # Why do we use switcher? 
        switcher = False
        for pred in converted_preds.data:
            if switcher is False:            
                preds.append([pred, 0])
            if switcher is True:
                preds[-1][1] = pred
            switcher = not switcher
        
        # Explanation of annotations --> http://human-pose.mpi-inf.mpg.de/#download
        # Use info about right hand and left hand 
        self.rhand = preds[10]
        self.lhand = preds[15]
        self.rshoulder = preds[12]
        self.lshoulder = preds[13]

        self.prediction_started = True; 

        if self.inspect_keypoints:  
            self.publish_predicted_keypoints(self.rhand, self.lhand)

    def calib_cb(self, msg): 
        
        self.start_calib = msg.data

        self.start_calib_time = rospy.Time.now().to_sec()

    def draw_zones_cb(self, stickman_img):

        rospy.logdebug("Entered stickman!")
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Rect for yaw
        draw.rectangle(self.ctl_zone, fill=(178,34,34, 100), width=2)       

        # Rect for height
        draw.rectangle(self.deadzone, fill=(178,34,34, 100), width=2)


        if self.hmi_compression: 
            rospy.loginfo("Compressing zones!")
            compressed_msg = convert_pil_to_ros_compressed(img, color_conversion="True")
            self.stickman_compressed_area_pub.publish(compressed_msg)            

        else:             
            rospy.loginfo("Publishing hmi control zones!")
            ros_msg = convert_pil_to_ros_img(img) 
            self.stickman_area_pub.publish(ros_msg)

        #if self.depth_recv: 
        #    
            # Test visual feedback
        #    d = self.relative_dist * 10
        #    pt1 = (self.rhand[0] - numpy.ceil(d), self.rhand[1] - numpy.ceil(d))
        #    pt2 = (self.rhand[0] + numpy.ceil(d), self.rhand[1] + numpy.ceil(d))
        #    rospy.logdebug("Current point: ({}, {})".format(pt1, pt2))
        #    draw.ellipse([pt1, pt2], fill=(0, 255, 0))

        #rospy.loginfo("stickman_cb duration is: {}".format(duration))
        
        duration = rospy.Time().now().to_sec() - start_time

    def depth_cb(self, msg): 
        
        #self.depth_msg = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(self.width, self.height, 4)
        self.depth_recv = False

    def depth_pcl_cb(self, msg): 

        #https://answers.ros.org/question/191265/pointcloud2-access-data/

        self.depth_pcl_recv = True
        self.depth_pcl_msg = PointCloud2()
        self.depth_pcl_msg = msg
   
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
        
if __name__ == '__main__':

    uC = uavController(sys.argv[1], sys.argv[2])
    uC.run()