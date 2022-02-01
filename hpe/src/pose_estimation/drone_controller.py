#!/opt/conda/bin/python3
from turtle import width
import rospy
import rospkg
import sys
import cv2
import numpy

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import Image, CompressedImage, Joy

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from hpe_ros_inference import HumanPoseEstimationROS

class uavController:

    def __init__(self, frequency):

        nn_init_time_sec = 10
        rospy.init_node("uav_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        self.current_x = 0
        self.current_y = 0
        self.current_z = 1
        self.current_rot = 0

        self.control_type = "joy" #position

        self._init_publishers(); self._init_subscribers(); 

        # Define zones
        self.height = 480; self.width = 640; 

        self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_rect_zones(self.width, self.height, 0.1, 0.05)

        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/hpe/include/arial.ttf", 20, encoding="unic")

        self.started = False
        self.rate = rospy.Rate(int(frequency))     

        self.inspect_keypoints = False
        self.recv_pose_meas = False

        self.hmi_integration = False

        self.initialized = True
        self.prediction_started = False

        rospy.loginfo("Initialized!")   


    def _init_publishers(self): 
        
        #TODO: Add topics to yaml file
        if self.control_type == "position":
            self.pose_pub = rospy.Publisher("bebop/pos_ref", Pose, queue_size=1)

        if self.control_type == "euler": 
            self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=1)

        self.stickman_area_pub = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)
        self.stickman_compressed_area_pub = rospy.Publisher("/stickman_compressed_ctl_area", CompressedImage, queue_size=1)

        self.lhand_x_pub = rospy.Publisher("hpe/lhand_x", Int32, queue_size=1)
        self.rhand_x_pub = rospy.Publisher("hpe/rhand_x", Int32, queue_size=1)
        self.lhand_y_pub = rospy.Publisher("hpe/lhand_y", Int32, queue_size=1)
        self.rhand_y_pub = rospy.Publisher("hpe/rhand_y", Int32, queue_size=1)

    def _init_subscribers(self): 

        self.preds_sub          = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        self.stickman_sub       = rospy.Subscriber("stickman", Image, self.draw_zones_cb, queue_size=1)
        self.current_pose_sub   = rospy.Subscriber("uav/pose", PoseStamped, self.curr_pose_cb, queue_size=1)
        
    def define_rect_zones(self, img_width, img_height, edge_offset, rect_width):
        
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
        roll_rect = ((cx + width_edge, cy-r_width), (img_width - width_edge), (cy + r_width))
        
        return height_rect, yaw_rect, pitch_rect, roll_rect
        
    def publish_predicted_keypoints(self, rhand, lhand): 

        rhand_x, rhand_y = rhand[0], rhand[1]; 
        lhand_x, lhand_y = lhand[0], lhand[1]

        rospy.logdebug("rhand: \t x: {}\t y: {}".format(rhand_x, rhand_y))
        rospy.logdebug("lhand: \t x: {}\t y: {}".format(lhand_x, lhand_y))

        self.lhand_x_pub.publish(int(lhand_x))
        self.lhand_y_pub.publish(int(lhand_y))
        self.rhand_x_pub.publish(int(rhand_x))
        self.rhand_y_pub.publish(int(rhand_y))

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
        
        # Use info about right hand and left hand 
        self.rhand = preds[10]
        self.lhand = preds[15]

        self.prediction_started = True; 

        if self.inspect_keypoints:  
            self.publish_predicted_keypoints(self.rhand, self.lhand)

    def draw_zones_cb(self, stickman_img):
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Rect for yaw
        draw.rectangle(self.yaw_rect, fill=(178,34,34, 100), width=2)       

        # Rect for height
        draw.rectangle(self.height_rect, fill=(178,34,34, 100), width=2)

        # Rectangles for pitch
        # draw.rectangle(self.pitch_rect, fill=(178,34,34, 100), width=2)
        
        # Rect for roll 
        # draw.rectangle(self.roll_rect, fill=(178,34,34, 100), width=2)
       
        # Text for changing UAV height and yaw
        offset_x = 2; offset_y = 2; 
        # Text writing
        ###########################################################################################################################################
        #up_size = uavController.get_text_dimensions("UP", self.font); down_size = uavController.get_text_dimensions("DOWN", self.font)
        #yp_size = uavController.get_text_dimensions("Y+", self.font); ym_size = uavController.get_text_dimensions("Y-", self.font)
        #draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - up_size[0]/2, self.height_area[0]- up_size[1] ), "UP", font=self.font, fill="black")
        #draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - down_size[0]/2, self.height_area[1]), "DOWN", font=self.font, fill="black")
        
        ############################################################################################################################################
        #draw.text(((self.rotation_area[0] - ym_size[0], (self.height_area[0] + self.height_area[1])/2 - ym_size[1]/2)), "Y-", font=self.font)
        #draw.text(((self.rotation_area[1], (self.height_area[0] + self.height_area[1])/2 - yp_size[1]/2)), "Y+", font=self.font)

        ######################################################################################################################################
        
        # Text for moving UAV forward and backward 
        #fwd_size = uavController.get_text_dimensions("FWD", self.font); bwd_size = uavController.get_text_dimensions("BWD", self.font)
        #l_size = uavController.get_text_dimensions("L", self.font); r_size = uavController.get_text_dimensions("R", self.font)
        #draw.text(((self.x_area[0] + self.x_area[1])/2 - r_size[0]/2, self.y_area[0] - r_size[1]), "L", font=self.font, fill="black")
        #draw.text(((self.x_area[0] + self.x_area[1])/2 - l_size[0]/2, self.y_area[1]), "R", font=self.font, fill="black")
        #draw.text(((self.x_area[0] - bwd_size[0], (self.y_area[0] + self.y_area[1])/2 - bwd_size[1]/2)), "BWD", font=self.font, fill="black")
        #draw.text(((self.x_area[1], (self.y_area[0] + self.y_area[1])/2 - fwd_size[1]/2)), "FWD", font=self.font, fill="black")
        ########################################################################################################################################

        # Check what this mirroring does here! --> mirroring is neccessary to see ourselves when operating 
        #rospy.loginfo("Publishing stickman with zones!")

        if self.hmi_integration: 
            rospy.loginfo("Compressing zones")
            compressed_msg = uavController.convert_pil_to_ros_compressed(img, color_conversion="True")
            self.stickman_compressed_area_pub.publish(compressed_msg)            

        else: 
            
            ros_msg = uavController.convert_pil_to_ros_img(img) 
            self.stickman_area_pub.publish(ros_msg)

        #TODO: Add compression (use image transport to add it? )

        duration = rospy.Time().now().to_sec() - start_time
        #rospy.loginfo("stickman_cb duration is: {}".format(duration))

    def run(self): 
        #rospy.spin()

        while not rospy.is_shutdown():
            if not self.initialized or not self.prediction_started: 
                rospy.sleep(0.1)
            else:

                # Reverse mirroring operation: 
                lhand_ = (abs(self.lhand[0] - self.width), self.lhand[1])
                rhand_ = (abs(self.rhand[0] - self.width), self.rhand[1])

                if self.control_type == "position": 

                    self.run_position_ctl(lhand_, rhand_)

                if self.control_type == "joy": 

                    self.run_joy_ctl(lhand_, rhand_)

                self.rate.sleep()

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

        cx = (x1-x0)/2 + x0; 
        cy = (y1-y0)/2 + y0; 

        return (cx, cy)

    def in_ctl_zone(self, point, rect, deadzone): 

        x, y = point[0], point[1]
        x0, y0  = rect[0][0], rect[0][1]
        x1, y1  = rect[1][0], rect[1][1]
        cx, cy = self.determine_center(rect)

        rect1 = ((x0, y0), (cx + deadzone, cy - deadzone) )
        rect2 = ((cx - deadzone, cy + deadzone), (x1, y1))

        # First rect 
        if self.in_zone(point, rect1) or self.in_zone(point, rect2): 

            norm_x_diff = (cx - x) / ((x1 - x0)/2)
            norm_y_diff = (cy - y) / ((y1 - y0)/2)

            return norm_x_diff, norm_y_diff

        else: 

            return 0.0, 0.0
        
    def run_position_ctl(self, lhand, rhand):

                # Convert predictions into drone positions. Goes from [1, movement_available]
        # NOTE: image is mirrored, so left control area in preds corresponds to the right hand movements 
        pose_cmd = Pose()        
        if self.recv_pose_meas and not self.started:
            rospy.logdebug("Setting up initial value!")
            pose_cmd.position.x = self.current_pose.pose.position.x
            pose_cmd.position.y = self.current_pose.pose.position.y
            pose_cmd.position.z = self.current_pose.pose.position.z
            pose_cmd.orientation.z = self.current_pose.pose.orientation.z

        elif self.recv_pose_meas and self.started: 
            try:
                pose_cmd = self.prev_pose_cmd  # Doesn't exist in the situation where we've started the algorithm however, never entered some of the zones!
            except:
                pose_cmd.position = self.current_pose.pose.position
                pose_cmd.orientation = self.current_pose.pose.orientation
        
        increase = 0.03; decrease = 0.03; 
        
        if self.started:
            self.changed_cmd = False
            # Current predictions
            rospy.logdebug("Left hand: {}".format(lhand))
            rospy.logdebug("Right hand: {}".format(rhand))
            

            if self.check_if_in_range(lhand[0], self.rotation_deadzone[0], self.rotation_deadzone[1]):                                      
                rospy.logdebug("Left hand inside of rotation area!")
                if self.check_if_in_range(lhand[1], self.height_area[0], self.height_deadzone[0]):
                    pose_cmd.position.z += increase
                    rospy.logdebug("Increasing z!")
                elif self.check_if_in_range(lhand[1], self.height_deadzone[1], self.height_area[1]):
                    pose_cmd.position.z -= decrease  
                    rospy.logdebug("Decreasing z!")
     
            #if self.check_if_in_range(lhand[1], self.height_area[0], self.height_area[1]):
            #    rospy.logdebug("Left hand inside of the height deadzone!")   
            #    if (self.check_if_in_range(lhand[0], self.rotation_deadzone[0], self.rotation_area[1])):
            #        pose_cmd.orientation.z += increase 
            # # TODO: Generate orientation correctly for quaternion
            #        rospy.logdebug("Increasing yaw!")
            #    
            #    elif(self.check_if_in_range(lhand[0], self.rotation_area[0], self.rotation_deadzone[1])):
            #        pose_cmd.orientation.z -= decrease
            #        rospy.logdebug("Decreasing yaw!")
            
            # Converter for x and y movements. Left hand is [15]   
            if self.check_if_in_range(rhand[0], self.x_area[0], self.x_area[1]):

                if (self.check_if_in_range(rhand[1], self.y_deadzone[1], self.y_area[1])):
                    pose_cmd.position.y -= increase
                    rospy.logdebug("Increasing y!")

                elif (self.check_if_in_range(rhand[1], self.y_area[0], self.y_deadzone[0])):
                    pose_cmd.position.y += decrease         
                    rospy.logdebug("Decreasing y!")   

            if self.check_if_in_range(rhand[1], self.y_area[0], self.y_area[1]):
                rospy.logdebug("Right hand inside of y area!")
                if self.check_if_in_range(rhand[0], self.x_deadzone[1], self.x_area[1]):
                    pose_cmd.position.x += increase    
                    rospy.logdebug("Increasing x!")

                elif self.check_if_in_range(rhand[0], self.x_area[0], self.x_deadzone[0]): 
                    pose_cmd.position.x -= decrease
                    rospy.logdebug("Decreasing x!")            
            

            rospy.loginfo("x:{0} \t y: {1} \t , z: {2} \t , rot: {3} \t".format(round(pose_cmd.position.x, 3),
                                                                                round(pose_cmd.position.y, 3),
                                                                                round(pose_cmd.position.z, 3),
                                                                                round(pose_cmd.orientation.z, 3)))
            self.prev_pose_cmd = pose_cmd
            self.pose_pub.publish(pose_cmd)


        # If not started yet, put both hand in the middle of the deadzones to start        
        else:
            # Good condition for starting 
            if rhand[0] > self.rotation_deadzone[0] and rhand[0] < self.rotation_deadzone[1] and rhand[1] > self.height_deadzone[0] and rhand[0] < self.height_deadzone[1]:
                if lhand[0] > self.x_deadzone[0] and lhand[1] < self.x_deadzone[1] and lhand[1] > self.y_deadzone[0] and lhand[1] < self.y_deadzone[1]:
                    rospy.loginfo("Started!")
                    self.started = True


        duration = rospy.Time.now().to_sec() - start_time
    
    def run_joy_ctl(self, lhand, rhand): 

        joy_msg = Joy()

        height_w, height_h = self.in_ctl_zone(lhand, self.height_rect, 20)
        height_cmd = height_h 

        yaw_w, yaw_h = self.in_ctl_zone(lhand, self.yaw_rect, 20)
        yaw_cmd = yaw_w

        rospy.logdebug("Height cmd: {}".format(height_cmd))
        rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))

    @staticmethod
    def convert_pil_to_ros_img(img):
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
    def convert_pil_to_ros_compressed(img, color_conversion = False, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 

        if color_conversion: 
            np_img = uavController.bgr2rgb(np_img)            
        
        compressed_img = cv2.imencode(".{}".format(compression_type), np_img)[1]
        msg.data = compressed_img.tobytes()

        return msg

    @staticmethod
    def get_text_dimensions(text_string, font):

        ascent, descent = font.getmetrics()
        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent

        return (text_width, text_height)
    
    @staticmethod
    def bgr2rgb(img):
        
        rgb_img = numpy.zeros_like(img)
        rgb_img[:, :, 0] = img[:, :, 2]
        rgb_img[:, :, 1] = img[:, :, 1]
        rgb_img[:, :, 2] = img[:, :, 0]
        
        return rgb_img

        

if __name__ == '__main__':

    uC = uavController(sys.argv[1])
    uC.run()