import numpy
import numpy as np

def replace_colot(img,src,target_array):
    r_img = img[:,:,0]
    g_img = img[:,:,1]
    b_img = img[:,:,2]

    img_array = r_img*256*256+g_img*256+b_img
    src_array = src[0]*256*256 + src[1]*256 +src[2]
    r_img[img_array==src_array]=target_array[0]
    g_img[img_array==src_array]=target_array[1]
    b_img[img_array==src_array]=target_array[2]
    dst_img = np.array([r_img,g_img,b_img],dtype=np.int32)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img
image_path = r'D:\2021\wwww\experiment\dual\result\CHASEDB1\dual_20_wei_1_2_ske_1_2_pcl_0.5_cop_1\199\colorImage_12R.jpg'
from PIL import Image
img = Image.open(image_path)
img  = np.array(img)
before = [0,255,0]
after= [255,255,0]
temp_img = replace_colot(img,before,after)
final_img = replace_colot(temp_img,[255,255,255],[0,255,0])
final_img = numpy.uint8(final_img)
final_img = Image.fromarray(final_img)

final_img.save('../predict_root/chasedb1.png')