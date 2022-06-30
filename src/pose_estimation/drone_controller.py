#!/opt/conda/bin/python3
import rospy
import rospkg
import sys
import cv2
import numpy 
import copy

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32, Float32, Bool
from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2
from visualanalysis_msgs.msg import BodyJoint2DArray
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
        rospy.init_node("uav_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        self._init_publishers();
        self._init_subscribers(); 

        # Define zones / dependent on input video image (Can be extracted from camera_info) 
        self.height     = 480; 
        self.width      = 640; 

        # 2D control zones
        self.ctl_width  = self.width/3.5; self.ctl_height = self.height/2
        self.r_zone     = self.define_ctl_zone( self.ctl_width, self.ctl_height, 3 * self.width/4, self.height/2)
        self.l_zone     = self.define_ctl_zone( self.ctl_width, self.ctl_height, self.width/4, self.height/2)

        rospy.logdebug("Right zone: {}".format(self.r_zone))
        rospy.logdebug("Left zone: {}".format(self.l_zone))

        self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_2d_ctl_zones(self.l_zone, self.r_zone, 25)

        # Define deadzones
        self.l_deadzone = self.define_deadzones(self.height_rect, self.yaw_rect)
        self.r_deadzone = self.define_deadzones(self.pitch_rect, self.roll_rect)

        self.start_position_ctl     = False
        self.start_joy_ctl          = False
        self.start_joy2d_ctl        = False

        # Debugging arguments
        self.inspect_keypoints  = False
        self.recv_pose_meas     = False        

        # Image compression for human-machine interface
        self.hmi_compression    = False

        # Flags for run method
        self.initialized        = True
        self.prediction_started = False
        self.stickman_published = False

        self.rate = rospy.Rate(int(frequency))     
        rospy.loginfo("Initialized!")   

    def _init_publishers(self): 
        
        self.joy_pub            = rospy.Publisher("/joy", Joy, queue_size=1)
        self.stickman_area_pub  = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)
        self.stickman_pub       = rospy.Publisher("/stickman", Image, queue_size=1)

    def _init_subscribers(self): 
        
        # Get human skeleton points
        self.preds_sub          = rospy.Subscriber("/uav/visualanalysis/human_pose_2d", BodyJoint2DArray, self.pred_cb, queue_size=1)
        self.cam_sub            = rospy.Subscriber("/uav/visualanalysis/rgb_camera", Image, self.img_cb, queue_size=1)
        self.stickman_sub       = rospy.Subscriber("stickman", Image, self.draw_zones_cb, queue_size=1)
        # It would be good to have method for itself here to be able to draw skeleton 
        self.current_pose_sub   = rospy.Subscriber("uav/pose", PoseStamped, self.curr_pose_cb, queue_size=1)
           
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
        yaw_rect    = ((width_edge, cy - r_width), (cx - width_edge, cy + r_width))
        # Define rectangle for pitch control
        pitch_rect  = ((cx2 - r_width, height_edge), (cx2 + r_width, img_height - height_edge))
        # Define rectangle for roll control 
        roll_rect   = ((cx + width_edge, cy-r_width), (img_width - width_edge, cy + r_width))
        
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
        cx, cy  = self.determine_center(rect)

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

    def run_position_ctl(self, lhand, rhand):

        # Convert predictions into drone positions. Goes from [1, movement_available]
        # NOTE: image is mirrored, so left control area in preds corresponds to the right hand movements 
        pose_cmd = Pose()        
        if self.recv_pose_meas and not self.start_position_ctl:
            rospy.logdebug("Setting up initial value!")
            pose_cmd.position.x = self.current_pose.pose.position.x
            pose_cmd.position.y = self.current_pose.pose.position.y
            pose_cmd.position.z = self.current_pose.pose.position.z
            pose_cmd.orientation.z = self.current_pose.pose.orientation.z

        elif self.recv_pose_meas and self.start_position_ctl: 
            try:
                pose_cmd = self.prev_pose_cmd  # Doesn't exist in the situation where we've started the algorithm however, never entered some of the zones!
            except:
                pose_cmd.position = self.current_pose.pose.position
                pose_cmd.orientation = self.current_pose.pose.orientation
        
        increase = 0.03; decrease = 0.03; 
        
        if self.start_position_ctl:
            self.changed_cmd = False
            # Current predictions
            rospy.logdebug("Left hand: {}".format(lhand))
            rospy.logdebug("Right hand: {}".format(rhand))
            
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

    # TODO: Fix this part!
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

        rospy.logdebug("x0: {}\t x1: {}".format(x0, x1))
        rospy.logdebug("y0: {}\t y1: {}".format(y0, y1))
        rospy.logdebug("cx: {}".format(cx))
        rospy.logdebug("cy: {}".format(cy))
        
        if self.in_zone(point, rect): 
            
            rospy.logdebug("x: {}".format(x))
            rospy.logdebug("y: {}".format(y))

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
            yaw_cmd *= reverse_dir
            pitch_cmd *= reverse_dir

        # Test!
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

    def run(self): 

        while not rospy.is_shutdown():
            if not self.initialized or not (self.prediction_started and self.stickman_published): 
                rospy.logdebug("Waiting prediction")
                rospy.sleep(0.1)
            else:
                
                # 9 and 10
                rhand_ = self.mirrored_preds[10]
                lhand_ = self.mirrored_preds[15]

                
                # Check start condition
                if self.in_zone(lhand_, self.l_deadzone) and self.in_zone(rhand_, self.r_deadzone):
                    self.start_joy2d_ctl = True 
                else:
                    rospy.loginfo("Not in deadzones!")        
                    
                if self.start_joy2d_ctl: 

                    self.run_joy2d_ctl(lhand_, rhand_)

                self.rate.sleep()

    def curr_pose_cb(self, msg):
        
        self.recv_pose_meas = True; 
        self.current_pose = PoseStamped(); 
        self.current_pose.header = msg.header
        self.current_pose.pose.position = msg.pose.position
        self.current_pose.pose.orientation = msg.pose.orientation

    def img_cb(self, msg): 

        rospy.loginfo("Recieved raw camera img.")

        # Convert ROS Image to PIL
        img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 

        # Plot stickman
        if self.prediction_started:
            points = self.hpe_preds

            self.mirrored_preds = mirror_points(points, 640)
            stickman_img = plot_stickman(img, self.mirrored_preds)

            # Convert to ROS msg
            stickman_msg = convert_pil_to_ros_img(stickman_img)

            # Publish stickman
            self.stickman_pub.publish(stickman_msg)

            self.stickman_published = True


    def pred_cb(self, preds):

        # Timestamp
        header = preds.header

        # Extract x,y coordinate for each joint from human pose estimation
        self.hpe_preds = list(map(extract_hpe_joint, preds.skeleton_2d)) 

        self.prediction_started = True


    def draw_zones_cb(self, stickman_img):
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # It seems like there's already mirroring
        # Mirror image here 
        # img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")

        # Control zones
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
            rospy.loginfo("Compressing zones")
            compressed_msg = convert_pil_to_ros_compressed(img, color_conversion="True")
            self.stickman_compressed_area_pub.publish(compressed_msg)            

        else:             
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

   
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def extract_hpe_joint(joint):

    return (joint.x, joint.y)

def mirror_points(points, width): 

    points_ = []
    for point in points:
        points_.append((abs(point[0] - width), point[1]))

    return points_ 

    
        
if __name__ == '__main__':

    uC = uavController(sys.argv[1], sys.argv[2])
    uC.run()