
#!/opt/conda/bin/python3
import numpy 
import rospy
import cv2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2

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

def convert_pil_to_ros_compressed(img, color_conversion = False, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 

        if color_conversion: 
            np_img = bgr2rgb(np_img)            
        
        compressed_img = cv2.imencode(".{}".format(compression_type), np_img)[1]
        msg.data = compressed_img.tobytes()

        return msg

def get_text_dimensions(text_string, font):

    ascent, descent = font.getmetrics()
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)
    
def bgr2rgb(img):
        
    rgb_img = numpy.zeros_like(img)
    rgb_img[:, :, 0] = img[:, :, 2]
    rgb_img[:, :, 1] = img[:, :, 1]
    rgb_img[:, :, 2] = img[:, :, 0]
        
    return rgb_img
