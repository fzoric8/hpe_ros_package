#!/opt/conda/bin/python3
import rospy
import sys
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from PIL import ImageDraw
from PIL import Image as PILImage
from PIL import ImageOps
import cv2
import numpy

class hpe_to_position:
    def __init__(self):
        self.current_x = 0
        self.current_y = 0
        self.current_z = 1
        self.current_rot = 0
        
        # Publishers and subscribers
        self.pose_pub = rospy.Publisher("uav/pose_ref", Pose, queue_size=1)
        self.preds_sub = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        self.stickman_sub = rospy.Subscriber("stickman", Image, self.stickman_cb, queue_size=1)
        self.stickman_area_pub = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)

        # Config for control areas
        self.height_area = [50, 350]
        self.height_deadzone = [180, 220]

        self.rotation_area = [30, 250]
        self.rotation_deadzone = [120, 160]

        
        self.x_area = [390, 610]
        self.x_deadzone = [480, 520]

        self.y_area = [50, 350]
        self.y_deadzone = [180, 220]

        self.started = False
        
        
    
    def pred_cb(self, converted_preds):
        preds = []
        switcher = False
        for pred in converted_preds.data:
            if switcher is False:            
                preds.append([pred, 0])
            if switcher is True:
                preds[-1][1] = pred
            switcher = not switcher
        
        # Convert predictions into drone positions. Goes from [1, movement_available]
        # NOTE: image is mirrored, so left control area in preds corresponds to the right hand movements
        pose = Pose()
        pose.position.z = 2
        pose.position.y = 0
        pose.position.x = 0
        pose.orientation.z = 0
        movement_available = 2  

        rhand = preds[10]
        lhand = preds[15]
        
        if self.started:
            # Converter for height and rotation. Right hand is [10]
            if rhand[0] > self.rotation_area[0] and rhand[0] < self.rotation_area[1]:                                       # If the hand is inside the green box
                if rhand[1] > self.height_deadzone[1] and rhand[1] < self.height_area[1]:                                   # If the hand is between the bottom green line and the bottom red deadzone line
                    pose.position.z = (self.height_area[1] - rhand[1]) / (self.height_area[1] - self.height_deadzone[1])    # Scale pixels to 1
                    pose.position.z = 1 + (pose.position.z * movement_available / 2)                                        # Additional scaling to fit [1, 1 + movement_available/2]

                elif rhand[1] < self.height_deadzone[0] and rhand[1] > self.height_area[0]:
                    pose.position.z = (rhand[1] - self.height_area[0]) / (self.height_deadzone[0] - self.height_area[0])
                    pose.position.z = 5 - (1 + (pose.position.z * movement_available / 2) + movement_available / 2)
                else:
                    pose.position.z = 2

                
            if rhand[1] > self.height_area[0] and rhand[1] < self.height_area[1]:
                if rhand[0] > self.rotation_deadzone[0] and rhand[0] < self.rotation_area[1]:
                    pose.orientation.z = (self.rotation_area[1] - rhand[0]) / (self.rotation_area[1] - self.rotation_deadzone[1])
                    pose.orientation.z = (pose.orientation.z * movement_available / 2)

                elif rhand[0] < self.rotation_deadzone[0] and rhand[0] > self.rotation_area[0]:
                    pose.orientation.z = (rhand[0] - self.rotation_area[0]) / (self.rotation_deadzone[0] - self.rotation_area[0])
                    pose.orientation.z = 4 - (1 + (pose.orientation.z * movement_available / 2) + movement_available / 2)
                else:
                    pose.orientation.z = 0


            # Converter for x and y movements. Left hand is [15]
            if lhand[0] > self.x_area[0] and lhand[0] < self.x_area[1]:
                if lhand[1] > self.y_deadzone[1] and lhand[1] < self.y_area[1]:
                    pose.position.y = (self.y_area[1] - lhand[1]) / (self.y_area[1] - self.y_deadzone[1])
                    pose.position.y = pose.position.y * movement_available
                    pose.position.y -= movement_available

                elif lhand[1] < self.y_deadzone[0] and lhand[1] > self.y_area[0]:
                    pose.position.y = (lhand[1] - self.y_area[0]) / (self.y_deadzone[0] - self.y_area[0])
                    pose.position.y = (pose.position.y * movement_available)
                    pose.position.y = movement_available - pose.position.y
            

            if lhand[1] > self.y_area[0] and lhand[1] < self.y_area[1]:
                if lhand[0] > self.x_deadzone[1] and lhand[1] < self.x_area[1]:           
                    pose.position.x = (self.x_area[1] - lhand[0]) / (self.x_area[1] - self.x_deadzone[1])
                    pose.position.x = pose.position.x * movement_available
                    pose.position.x -= movement_available

                elif lhand[0] < self.x_deadzone[0] and lhand[0] > self.x_area[0]:
                    pose.position.x = (lhand[0] - self.x_area[0]) / (self.x_deadzone[0] - self.x_area[0])
                    pose.position.x = (pose.position.x * movement_available)
                    pose.position.x = movement_available - pose.position.x


            pose.orientation.z = 0
            print("x, y, z, rot:" + str(pose.position.x) + "," + str(pose.position.y) + "," + str(pose.position.z) + "," + str(pose.orientation.z))
            self.pose_pub.publish(pose)


        # If not started yet, put both hand in the middle of the deadzones to start        
        else:
            self.pose_pub.publish(pose)
            if rhand[0] > self.rotation_deadzone[0] and rhand[0] < self.rotation_deadzone[1] and rhand[1] > self.height_deadzone[0] and rhand[0] < self.height_deadzone[1]:
                if lhand[0] > self.x_deadzone[0] and lhand[1] < self.x_deadzone[1] and lhand[1] > self.y_deadzone[0] and lhand[1] < self.y_deadzone[1]:
                    print("Started!")
                    self.started = True
        
     
    def stickman_cb(self, stickman_img):
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img)
        
        # Rectangles for height and rotation
        draw.rectangle([(self.rotation_area[0], self.height_deadzone[0]), (self.rotation_area[1], self.height_deadzone[1])], outline ="red", width=2)
        draw.rectangle([(self.rotation_deadzone[0], self.height_area[0]), (self.rotation_deadzone[1], self.height_area[1])], outline ="red", width=2)
        draw.rectangle([(self.rotation_area[0], self.height_area[0]), (self.rotation_area[1], self.height_area[1])], outline ="green", width=2)

        # Rectangles for movement left-right and forward-backward
        draw.rectangle([(self.x_area[0], self.y_deadzone[0]), (self.x_area[1], self.y_deadzone[1])], outline ="red", width=2)
        draw.rectangle([(self.x_deadzone[0], self.y_area[0]), (self.x_deadzone[1], self.y_area[1])], outline ="red", width=2)
        draw.rectangle([(self.x_area[0], self.y_area[0]), (self.x_area[1], self.y_area[1])], outline ="green", width=2)

        #draw.rectangle([(80, 180), (200, 220)], outline ="red", width=2)
        #draw.rectangle([(80, 50), (200, 350)], outline ="green", width=2)

        img = ImageOps.mirror(img)        
        ros_msg = hpe_to_position.convert_pil_to_ros_img(img)
        self.stickman_area_pub.publish(ros_msg)
    
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
        

rospy.init_node("hpe_to_position")
HtP = hpe_to_position()
rospy.spin()
