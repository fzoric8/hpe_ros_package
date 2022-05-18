#!/opt/conda/bin/python3
from audioop import avg
from multiprocessing import reduction
import queue
from turtle import width
import rospy
import rospkg
import sys
import cv2
import numpy as np
import math
import copy

from geometry_msgs.msg import Pose, PoseStamped, Vector3, Quaternion
from std_msgs.msg import Float64MultiArray, Int32, Float32, Bool
from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
import sensor_msgs.point_cloud2 as pc2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from img_utils import *
from drone import Drone, FlightMode
#from tasks import PreFlight

# TODO:
# - think of behavior when arm goes out of range! --> best to do nothing, just send references when there's signal from HPE 
# - add calibration method
# - decouple depth and zone calibration 
# - test 2D zones


class uavController:

    def __init__(self, frequency, control_type):

        nn_init_time_sec = 10
        rospy.init_node("uav_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        # Available control types are: euler, euler2d 
        if control_type: 
            self.control_type =  control_type
        else: 
            self.control_type = "euler2d"
        # Copy control type    
        self.init_control_type = copy.deepcopy(self.control_type)

        self._init_publishers();
        self._init_subscribers(); 
        self._init_service_proxys(); 

        # Define zones / dependent on input video image (Can be extracted from camera_info) 
        self.height = 480; 
        self.width = 640; 

        # Decoupled control  
        if self.control_type == "euler": 
            
            # 1D control zones
            self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_ctl_zones(self.width, self.height, 0.2, 0.05)

        # Coupled control --> Yaw/Height on same axis, Roll/Pitch on same axis
        if self.control_type == "euler2d": 

            # 2D control zones
            self.ctl_width = self.width/3.5; self.ctl_height = self.height/2
            self.r_zone = self.define_ctl_zone( self.ctl_width, self.ctl_height, 3 * self.width/4, self.height/2)
            self.l_zone = self.define_ctl_zone( self.ctl_width, self.ctl_height, self.width/4, self.height/2)

            rospy.logdebug("Right zone: {}".format(self.r_zone))
            rospy.logdebug("Left zone: {}".format(self.l_zone))

            self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_2d_ctl_zones(self.l_zone, self.r_zone, 25)

        # Define deadzones
        self.l_deadzone = self.define_deadzones(self.height_rect, self.yaw_rect)
        self.r_deadzone = self.define_deadzones(self.pitch_rect, self.roll_rect)

        self.start_position_ctl = False
        self.start_joy_ctl = False
        self.start_joy2d_ctl = False
        
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
        self.first_position_ctl = True

        # EPFL 
        self.yaw_cmd    = 0; 
        self.height_cmd = 0; 
        self.roll_cmd   = 0; 
        self.pitch_cmd  = 0; 
        self.pose_cmd = PoseStamped()
        self.current_pose = PoseStamped()
        init_position = Vector3(0,0,0)
        init_orientation = Quaternion(0, 0, 0, 1)
        self.current_pose.pose.position = init_position
        self.current_pose.pose.orientation = init_orientation
        self.drone = Drone("drone_1", False, True, 'local')
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

        # EPFL PUB
        #self.pose_cmd_pub = rospy.Publisher("/drone_1/mavros/setpoint_position/local", PoseStamped, queue_size=1)

    def _init_subscribers(self): 

        self.preds_sub          = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)

        # TODO: Add current pose from PX4
        # /ns/mavros/position_src/pose
        # position_src -> loca_position or vision_pose
        self.current_pose_sub   = rospy.Subscriber("/drone_1/mavros/local_position/pose", PoseStamped, self.curr_pose_cb, queue_size=1)

        # stickman 
        self.stickman_sub       = rospy.Subscriber("stickman", Image, self.draw_zones_cb, queue_size=1)
        
        # Add mavros state sub
        # self.uav_state_sub = rospy.Subscriber("drone_1/mavros/state", State, self.uav_state_cb)

    def _init_service_proxys(self): 

        # Services
        self.arming_client = rospy.ServiceProxy('drone_1/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('drone1/mavros/set_mode', SetMode)
           
    def publish_predicted_keypoints(self, rhand, lhand): 

        rhand_x, rhand_y = rhand[0], rhand[1]; 
        lhand_x, lhand_y = lhand[0], lhand[1]

        rospy.logdebug("rhand: \t x: {}\t y: {}".format(rhand_x, rhand_y))
        rospy.logdebug("lhand: \t x: {}\t y: {}".format(lhand_x, lhand_y))

        self.lhand_x_pub.publish(int(lhand_x))
        self.lhand_y_pub.publish(int(lhand_y))
        self.rhand_x_pub.publish(int(rhand_x))
        self.rhand_y_pub.publish(int(rhand_y))

    def define_ctl_zones(self, img_width, img_height, edge_offset, rect_width):
        
        # img center
        cx, cy = img_width/2, img_height/2
        # 1st zone
        cx1, cy1 = cx/2, cy/2
        # 2nd zone
        cx2, cy2 = cx + cx1, cy + cy1
        
        # Define offsets from edge
        if edge_offset < 1: 
            height_edge = edge_offset * img_height
            width_edge = edge_offset/2 * img_width

        # Zone definition 
        if rect_width < 1: 
            r_width = rect_width * img_width

        # Define rectangle for height control
        height_rect = ((cx1 - r_width, height_edge), (cx1 + r_width, img_height - height_edge))
        # Define rectangle for yaw control
        yaw_rect = ((width_edge, cy - r_width), (cx - width_edge, cy + r_width))
        # Define rectangle for pitch control
        pitch_rect = ((cx2 - r_width, height_edge), (cx2 + r_width, img_height - height_edge))
        # Define rectangle for roll control 
        roll_rect = ((cx + width_edge, cy-r_width), (img_width - width_edge, cy + r_width))
        
        return height_rect, yaw_rect, pitch_rect, roll_rect

    def define_calibrated_ctl_zones(self, calibration_points, img_w, img_h, w_perc=0.2, h_perc=0.3, rect_perc=0.05):

        cx1, cy1 = calibration_points[0][0], calibration_points[0][1]
        cx2, cy2 = calibration_points[1][0], calibration_points[1][1]

        # main_control dimensions
        a = img_w * w_perc
        b = img_h * h_perc

        # control_rect width
        c = img_w * rect_perc
        d = img_h * rect_perc

        # Define rectangles for heigh, yaw, pitch and roll
        height_rect     = ((cx1 - c, cy1 - b), (cx1 + c, cy1 + b))
        yaw_rect        = ((cx1 - a, cy1 - d), (cx1 + a, cy1 + d))
        pitch_rect      = ((cx2 - c, cy2 - b), (cx2 + c, cy2 + b))
        roll_rect       = ((cx2 - a, cy2 - d), (cx2 + a, cy2 + d))

        return height_rect, yaw_rect, pitch_rect, roll_rect    
    
    def define_deadzones(self, rect1, rect2):

        # valid if first rect vertical, second rect horizontal
        x01, y01 = rect1[0][0], rect1[0][1]
        x11, y11 = rect1[1][0], rect1[1][1]

        x02, y02 = rect2[0][0], rect2[0][1]
        x12, y12 = rect2[1][0], rect2[1][1]

        deadzone_rect = ((x01, y02), (x11, y12))

        return deadzone_rect      

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

    def determine_center(self, rect):

        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]

        cx = (x1 - x0) / 2 + x0; 
        cy = (y1 - y0) / 2 + y0; 

        return (cx, cy)

    def in_ctl_zone(self, point, rect, deadzone, orientation): 

        x, y = point[0], point[1]
        x0, y0  = rect[0][0], rect[0][1]
        x1, y1  = rect[1][0], rect[1][1]
        cx, cy = self.determine_center(rect)

        if orientation == "vertical":
            rect1 = ((x0, y0), (cx + deadzone, cy - deadzone) )
            rect2 = ((cx - deadzone, cy + deadzone), (x1, y1))

        if orientation == "horizontal": 

            rect1 = ((x0, y0), (cx - deadzone, cy + deadzone))
            rect2 = ((cx + deadzone, cy - deadzone), (x1, y1))

        # Check in which rect is point located
        if self.in_zone(point, rect1) or self.in_zone(point, rect2): 

            norm_x_diff = (cx - x) / ((x1 - x0)/2)
            norm_y_diff = (cy - y) / ((y1 - y0)/2)

            return norm_x_diff, norm_y_diff

        else: 

            return 0.0, 0.0

    def compose_joy_msg(self, pitch, roll, yaw, height):

        joy_msg = Joy()

        joy_msg.header.stamp = rospy.Time.now()
        joy_msg.axes = [yaw, height, roll, pitch]
        joy_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        return joy_msg

    def run_2d_position_ctl(self):

        if self.first_position_ctl:
            self.pose_cmd.pose.position.x = self.current_pose.pose.position.x 
            self.pose_cmd.pose.position.y = self.current_pose.pose.position.y
            self.pose_cmd.pose.position.z = self.current_pose.pose.position.z 
            self.pose_cmd.pose.orientation = self.current_pose.pose.orientation

            qx = self.current_pose.pose.orientation.x
            qy = self.current_pose.pose.orientation.y 
            qz = self.current_pose.pose.orientation.z 
            qw = self.current_pose.pose.orientation.w 

            roll, pitch, self.yaw = quat_to_eul_conv(qx, qy, qz, qw)

        else:
            
            yaw = self.yaw
            s_ = 0.01; 
            yaw = yaw + self.yaw_cmd * s_ * 5; self.yaw = yaw
            x_ = self.roll_cmd * s_* np.sin(yaw) - self.pitch_cmd * s_ * np.cos(yaw)
            y_ = self.pitch_cmd * s_ * np.sin(yaw) + self.roll_cmd * s_ * np.cos(yaw)

            # Dummy check to prevent unwanted reference jumps
            if x_ > 0.1: x_ = 0; 
            if y_ > 0.1: y_ = 0;
        
            self.pose_cmd.pose.position.x += x_; 
            self.pose_cmd.pose.position.y += y_; 
            self.pose_cmd.pose.position.z += self.height_cmd * s_; 
            qw_, qx_, qy_, qz_ = toQuaternion(yaw, 0, 0)
            self.pose_cmd.pose.orientation.x = qx_; self.pose_cmd.pose.orientation.y = qy_; 
            self.pose_cmd.pose.orientation.z = qz_; self.pose_cmd.pose.orientation.w = qw_; 
        
        #self.pose_cmd_pub.publish(self.pose_cmd)
        self.first_position_ctl = False
            
    # TODO: Implement position change in same way it has been implemented for joy control     
    def run_joy_ctl(self, lhand, rhand): 

        joy_msg = Joy()

        height_w, height_h = self.in_ctl_zone(lhand, self.height_rect, 20, orientation="vertical")
        height_cmd = height_h 

        yaw_w, yaw_h = self.in_ctl_zone(lhand, self.yaw_rect, 20, orientation="horizontal")
        yaw_cmd = yaw_w

        pitch_w, pitch_h = self.in_ctl_zone(rhand, self.pitch_rect, 20, orientation="vertical")
        pitch_cmd = pitch_h

        roll_w, roll_h = self.in_ctl_zone(rhand, self.roll_rect, 20, orientation="horizontal")
        roll_cmd = roll_w 

        reverse_dir = -1
        # Added reverse because rc joystick implementation has reverse
        reverse = True 
        if reverse: 
            roll_cmd  *= reverse_dir

        rospy.logdebug("Height cmd: {}".format(height_cmd))
        rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))
        rospy.logdebug("Pitch cmd: {}".format(pitch_cmd))
        rospy.logdebug("Roll cmd: {}".format(roll_cmd))
        
        # Compose from commands joy msg
        joy_msg = self.compose_joy_msg(pitch_cmd, roll_cmd, yaw_cmd, height_cmd)

        # Publish composed joy msg
        self.joy_pub.publish(joy_msg)
    
    # 2D control 
    def define_ctl_zone(self, w, h, cx, cy):

        px1 = cx - w/2; px2 = cx + w/2
        py1 = cy - h/2; py2 = cy + h/2

        # Conditions to contain control zone in image
        if px1 < 0: 
            px1 = 0        
        if py1 < 0:
            py1 = 0
        if px2 > self.width: 
            px2 = self.width        
        if py2 > self.height: 
            py2 = self.height

        ctl_rect = ((px1, py1), (px2, py2))

        return ctl_rect 

    def define_2d_ctl_zones(self, l_zone, r_zone, deadzone): 

        cx1, cy1 = (l_zone[0][0] + l_zone[1][0])/2, (l_zone[0][1] + l_zone[1][1])/2
        cx2, cy2 = (r_zone[0][0] + r_zone[1][0])/2, (r_zone[0][1] + r_zone[1][1])/2

        rospy.logdebug("cx1, cy1: {}, {}".format(cx1, cy1))
        rospy.logdebug("cx2, cy2: {}, {}".format(cx2, cy2))
        
        height_rect = ((cx1 - deadzone, l_zone[0][1]), (cx1 + deadzone, l_zone[1][1]))
        pitch_rect = ((cx2 - deadzone, r_zone[0][1]), (cx2 + deadzone, r_zone[1][1]))

        roll_rect = ((r_zone[0][0], cy2 - deadzone), (r_zone[1][0], cy2 + deadzone))
        yaw_rect = ((l_zone[0][0], cy1 - deadzone), (l_zone[1][0], cy1 + deadzone))

        rospy.logdebug("Height: {}".format(height_rect))
        rospy.logdebug("Pitch: {}".format(pitch_rect))
        rospy.logdebug("Roll: {}".format(roll_rect))
        rospy.logdebug("Yaw: {}".format(yaw_rect))

        return height_rect, yaw_rect, pitch_rect, roll_rect

    def in_ctl2d_zone(self, point, rect, deadzone): 

        x, y = point[0], point[1]
        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]
        cx, cy = (x1 + x0) / 2, (y1 + y0) / 2

        #rospy.logdebug("x0: {}\t x1: {}".format(x0, x1))
        #rospy.logdebug("y0: {}\t y1: {}".format(y0, y1))
        #rospy.logdebug("cx: {}".format(cx))
        #rospy.logdebug("cy: {}".format(cy))
        
        if self.in_zone(point, rect): 
            
            #rospy.logdebug("x: {}".format(x))
            #rospy.logdebug("y: {}".format(y))

            if abs(cx - x) > deadzone: 
                norm_x_diff = (x - cx) / ((x1 - x0) / 2)
            else: 
                norm_x_diff = 0.0

            if abs(cy - y) > deadzone: 
                norm_y_diff = (y - cy) / ((y1 - y0) / 2)
            else: 
                norm_y_diff = 0.0
        
        else: 

            norm_x_diff, norm_y_diff = 0.0, 0.0

        return norm_x_diff, norm_y_diff

    def run_joy2d_ctl(self, lhand, rhand): 

        yaw_cmd, height_cmd = self.in_ctl2d_zone(lhand, self.l_zone, 25)
        roll_cmd, pitch_cmd = self.in_ctl2d_zone(rhand, self.r_zone, 25)

        reverse_dir = -1
        # Added reverse because rc joystick implementation has reverse
        reverse = True 
        if reverse: 
            height_cmd *= reverse_dir
            #yaw_cmd *= reverse_dir
            #pitch_cmd  *= reverse_dir

        # Test!
        rospy.logdebug("Height cmd: {}".format(height_cmd))
        rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))
        rospy.logdebug("Pitch cmd: {}".format(pitch_cmd))
        rospy.logdebug("Roll cmd: {}".format(roll_cmd))

        # Set new commands into class_variable 
        self.height_cmd = height_cmd
        self.yaw_cmd = yaw_cmd
        self.pitch_cmd = pitch_cmd
        self.roll_cmd = roll_cmd

        # Compose from commands joy msg
        joy_msg = self.compose_joy_msg(pitch_cmd, roll_cmd, yaw_cmd, height_cmd)

        # Publish composed joy msg
        self.joy_pub.publish(joy_msg)

    def run_joy2d_dpth_ctl(self, lhand, rhand, d): 

        yaw_cmd, height_cmd = self.in_ctl2d_zone(lhand, self.l_zone, 25)
        pitch_cmd, roll_cmd = self.in_ctl2d_zone(rhand, self.r_zone, 25)

        # Depth cmd
        # pitch_cmd = curr_depth - avg_depth 
        # Use relative depth as pitch command
        pitch_cmd = d

        reverse_dir = -1
        # Added reverse because rc joystick implementation has reverse
        reverse = True 
        if reverse: 
            roll_cmd  *= reverse_dir

        # Test!
        debug_joy = True
        if debug_joy:
            rospy.logdebug("Height cmd: {}".format(height_cmd))
            rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))
            rospy.logdebug("Pitch cmd: {}".format(pitch_cmd))
            rospy.logdebug("Roll cmd: {}".format(roll_cmd))

        # Compose from commands joy msg
        joy_msg = self.compose_joy_msg(pitch_cmd, roll_cmd, yaw_cmd, height_cmd)

        # Publish composed joy msg
        self.joy_pub.publish(joy_msg)

    def zones_calibration(self, right, left, done):
        
        if not done: 
            self.rhand_calib_px.append(right[0]), self.rhand_calib_py.append(right[1])
            self.lhand_calib_px.append(left[0]), self.lhand_calib_py.append(left[1])
        
        else: 

            avg_rhand = (int(sum(self.rhand_calib_px)/len(self.rhand_calib_px)), int(sum(self.rhand_calib_py)/len(self.rhand_calib_py)))
            avg_lhand = (int(sum(self.lhand_calib_py)/len(self.lhand_calib_px)), int(sum(self.lhand_calib_py)/len(self.lhand_calib_py)))

            return avg_rhand, avg_lhand

    def create_float32_msg(self, value):

        msg = Float32()

        if value:  
            msg.data = value
        else:
            msg.data = -1.0 

        return msg

    def run(self): 

        while not rospy.is_shutdown():
            if not self.initialized or not self.prediction_started: 
                rospy.logdebug("Waiting prediction")
                rospy.sleep(0.1)
            else:

                # Reverse mirroring operation: 
                lhand_ = (abs(self.lhand[0] - self.width), self.lhand[1])
                rhand_ = (abs(self.rhand[0] - self.width), self.rhand[1])
                lshoulder_ = (abs(self.lshoulder[0] - self.width), self.lshoulder[1])
                rshoulder_ = (abs(self.rshoulder[0] - self.width), self.rshoulder[1])

                # ====================== Execution ========================
                if self.control_type == "position": 

                    self.run_position_ctl(lhand_, rhand_)

                if self.control_type == "euler":                

                    if self.in_zone(lhand_, self.l_deadzone) and self.in_zone(rhand_, self.r_deadzone):
                        self.start_joy_ctl = True 
                    else:
                        rospy.loginfo("Not in deadzones!")           
                    
                    if self.start_joy_ctl:
                        self.run_joy_ctl(lhand_, rhand_)

                if self.control_type == "euler2d":                                 
                        
                    if self.in_zone(lhand_, self.l_deadzone) and self.in_zone(rhand_, self.r_deadzone):
                        
                        t0 = rospy.get_time()
                        while self.drone.alive:
                            self.drone.set_target_pose(self.current_pose)
                            self.drone.set_mode(FlightMode.Offboard)
                            self.drone.arm()
                            self.drone.sleep()                    

                            if self.drone.armed and self.drone.mode == FlightMode.Offboard.value:
                                self.start_joy2d_ctl = True 
                                break    

                            if rospy.get_time() - t0 >= 15:
                                rospy.logerr('Preflight timeout, quitting...')
                                if self.drone.armed:
                                    self.drone.disarm()
                                rospy.signal_shutdown('Preflight timeout')
                                sys.exit(1)
                        
                    else:

                        rospy.logdebug("Arms are not in control zones!")                         
                    
                    if self.start_joy2d_ctl: 
                        self.run_joy2d_ctl(lhand_, rhand_)
                        self.run_2d_position_ctl()
                        self.drone.set_target_pose(self.pose_cmd)

                        # Add stuff for sending referencess

                        # Publish new pose

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


    def draw_zones_cb(self, stickman_img):

        #rospy.logdebug("Entered stickman!")
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")

        if self.control_type == "euler2d": 

            draw.rectangle(self.l_zone, width = 3)
            draw.rectangle(self.r_zone, width = 3)
        
        # Rect for yaw
        draw.rectangle(self.yaw_rect, fill=(178,34,34, 100), width=2)       

        # Rect for height
        draw.rectangle(self.height_rect, fill=(178,34,34, 100), width=2)

        # Rectangles for pitch
        # Pitch rect not neccessary when controlling depth
        draw.rectangle(self.pitch_rect, fill=(178,34,34, 100), width=2)
        
        # Rect for roll 
        draw.rectangle(self.roll_rect, fill=(178,34,34, 100), width=2)

        if self.hmi_compression: 
            rospy.loginfo("Compressing zones!")
            compressed_msg = convert_pil_to_ros_compressed(img, color_conversion="True")
            self.stickman_compressed_area_pub.publish(compressed_msg)            

        else:             
            #rospy.loginfo("Publishing hmi control zones!")
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

def quat_to_eul_conv( qx, qy, qz, qw):
    """
    Convert quaternions to euler angles (roll, pitch, yaw)
    """

    # roll (x-axis rotation)
    sinr = 2. * (qw * qx +  qy * qz)
    cosr = 1. - 2. * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    # pitch (y-axis rotation)
    sinp = 2. * (qw * qy - qz * qx)
    sinp = 1. if sinp > 1. else sinp
    sinp = -1. if sinp < -1. else sinp
    pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny = 2. * (qw * qz + qx * qy)
    cosy = 1 - 2. * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw

def toQuaternion(yaw, pitch,  roll):

    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);

    qw = cr * cp * cy + sr * sp * sy;
    qx = sr * cp * cy - cr * sp * sy;
    qy = cr * sp * cy + sr * cp * sy;
    qz = cr * cp * sy - sr * sp * cy;

    return qw, qx, qy, qz


   
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
        
if __name__ == '__main__':

    uC = uavController(sys.argv[1], sys.argv[2])
    uC.run()