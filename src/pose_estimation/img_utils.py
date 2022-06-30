
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

def plot_stickman(img, points, point_width=4): 

    plot_joint_dot = True; 
    plot_joint_num = True; 
    plot_stickman_ = True; 

    draw = ImageDraw.Draw(img); 

    # get a font
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)

    for i, joint in enumerate(points): 
        #draw.point((joint[0], joint[1]), width=2)
        r=5
        if plot_joint_dot: 
            draw.ellipse((joint[0]-r, joint[1]-r, joint[0]+r, joint[1]+r), fill=(255, 0, 0))
        if plot_joint_num:
            draw.text((joint[0], joint[1]), "{}".format(i),  fill=(0, 155, 0, 255), font=fnt)

    if plot_stickman_: 
        check_conditions_and_draw_lines(draw, points)

    return img

def check_conditions_and_draw_lines(draw, points): 

    if numpy.all(points[16] != [-1, -1]) and numpy.all(points[14] != [-1, -1]):
        plot_line(draw, points, 16, 14, color=(128, 0, 0))
    if numpy.all(points[12] != [-1, -1]) and numpy.all(points[12] != [-1, -1]): 
        plot_line(draw, points, 14, 12, color=(128, 0, 0))
    if numpy.all(points[12] != [-1, -1]) and numpy.all(points[6] != [-1, -1]):
        plot_line(draw, points, 12, 6)
    if numpy.all(points[6] != [-1, -1]) and numpy.all(points[8] != [-1, -1]):
        plot_line(draw, points, 6, 8)
    if numpy.all(points[8] != [-1, -1]) and numpy.all(points[10] != [-1, -1]):
        plot_line(draw, points, 8, 10)
    if numpy.all(points[6] != [-1, -1]) and numpy.all(points[5] != [-1, -1]):
        plot_line(draw, points, 6, 5)
    if numpy.all(points[11] != [-1, -1]) and numpy.all(points[13] != [-1, -1]):
        plot_line(draw, points, 11, 13)
    if numpy.all(points[13] != [-1, -1]) and numpy.all(points[15] != [-1, -1]):
        plot_line(draw, points, 13, 15)
    if numpy.all(points[0] != [-1, -1]) and numpy.all(points[2] != [-1, -1]):
        plot_line(draw, points, 0, 2)
    if numpy.all(points[11] != [-1, -1]) and numpy.all(points[5] != [-1, -1]):
        plot_line(draw, points, 11, 5)
    if numpy.all(points[5] != [-1, -1]) and numpy.all(points[7] != [-1, -1]):
        plot_line(draw, points, 5, 7)
    if numpy.all(points[7] != [-1, -1]) and numpy.all(points[9] != [-1, -1]):
        plot_line(draw, points, 7, 9)
    if numpy.all(points[1] != [-1, -1]) and numpy.all(points[0] != [-1, -1]):
        plot_line(draw, points, 1, 0)
    if numpy.all(points[1] != [-1, -1]) and numpy.all(points[3] != [-1, -1]):
        plot_line(draw, points, 1, 3)
    if numpy.all(points[2] != [-1, -1]) and numpy.all(points[4] != [-1, -1]):
        plot_line(draw, points, 2, 4)

    


def plot_line(draw, points, idx, idy, color=128): 

    draw.line((points[idx][0], points[idx][1] , points[idy][0], points[idy][1]), fill=color, width=3)

    
def bgr2rgb(img):
        
    rgb_img = numpy.zeros_like(img)
    rgb_img[:, :, 0] = img[:, :, 2]
    rgb_img[:, :, 1] = img[:, :, 1]
    rgb_img[:, :, 2] = img[:, :, 0]
        
    return rgb_img
