import pyrealsense2 as rs #摄像头的库
import numpy as np #数组和矩阵运算的库
import cv2 #图像处理库
pipeline = rs.pipeline()
config = rs.config()
#获取深度图、彩色图，图片尺寸424*240，帧率30
config.enable_stream(rs.stream.depth,424,240,rs.format.z16,30)
config.enable_stream(rs.stream.color,424,240,rs.format.bgr8,30)
#将深度图和彩色图对齐
align_to = rs.stream.depth
alignedFs = rs.align(align_to)
profile = pipeline.start(config)
def Change(position,i):    
    pos_1[i,0]=position[i,0]-212
    pos_1[i,1]=120-position[i,1]
    pos_1[i,2]=position[i,2]-length
    pos_2[i,0]=position[i,2]*pos_1[i,0]/length    
    pos_2[i,1]=position[i,2]*pos_1[i,1]/length
    pos_2[i,2]=pos_1[i,2]    
    pos_3[i,0]=pos_2[i,0]
    pos_3[i,1]=pos_2[i,2]
    pos_3[i,2]=pos_2[i,1]
    return pos_3
#主函数
try:
    while True:
        # 等待一对连续的帧（包含深度和颜色）
        frames = pipeline.wait_for_frames()  
        aligned_frames = alignedFs.process(frames)
        #对齐彩色图和深度图
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        #将图片转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #过滤掉太远和太近的数据
        depth_image[depth_image>250]=0#距离大于25公分的数据变为0
        depth_image[depth_image<150]=0#距离小于15公分的数据变为0
        #更改数据类型
        depth_image = np.array(depth_image,dtype='uint8')
        color_image = np.array(color_image,dtype='uint8')    
        #将色彩图转换为灰度图
        color_gray=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        #创建4个4*3的数组，所有元素都为0
        pos=np.zeros([4,3])
        pos_1=np.zeros([4,3])
        pos_2=np.zeros([4,3])
        pos_3=np.zeros([4,3])
    #获取特征点坐标
        #备份深度图，对备份图进行二值化
        depth_copy=depth_image.copy()
        depth_copy[depth_copy>0]=1
        #将灰度图和二值化深度图相乘，过滤掉灰度图中距离过远和过近的点（将值设为0）
        color_gray1=color_gray*depth_copy
        #将被过滤掉的点的值从0变为255
        color_gray1[color_gray1==0]=255
        #对图片进行降噪，使用的高斯模糊，卷积核为5*5
        color_gray_gs = cv2.GaussianBlur(color_gray1,(5,5),0)
        #寻找灰度图全局灰度极值点，返回极小值点坐标，即为特征点坐标
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(color_gray_gs)
        p1gray=minLoc[0]
        p0gray=minLoc[1]
        pos[0,0]=minLoc[1]#记录下特征点X坐标
        pos[0,1]=minLoc[0]#记录下特征点Y坐标
        #通过与深度图对应，获得特征点Z坐标
        pos[0,2]=depth_image[p0gray,p1gray]
        #在图片中标记出极小值点位置，并存储
        cv2.circle(color_gray1, minLoc, 7, (255, 0, 0), 2)
        cv2.imwrite("e:\\picture\\color_gray_circle.png",color_gray1)    
        #坐标变换
        length=201
        Change(pos,0)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally: 
    pipeline.stop() #结束流
