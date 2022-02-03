import os
import shutil

class Global(object):
    base_save_path = './output/video'
    rtsp_updated = False
    global_imgs = None
    global_source_file = None
    info_for_sort = [] #表示可以接收5个rtsp
    imgs_all_rtsp = []
    rtsp_list = []
    save_list = []
    rtsp_nums = None  #当前为2
    rtsp_updated =False
    mutex = None
    count = 0
    '''使用全局的计数来标志，不能用局部变量'''
    tizi_illegal_times = 0
    gaodu_illegal_times = 0
    # project_base = '/home/lihao/udisk/sourcecode/detect_track_project_basic_nosave_end1_struct_ubuntu_tizi'
    def init_list(self):
        if self.rtsp_nums:
            for i in range(self.rtsp_nums):
                self.info_for_sort[i] = None

def mkdir(dir):
    os.makedirs(dir, exist_ok=True)

def remkdir(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)
# class Global(object):
#     def __init__(self):
#         self.global_imgs = None
#         self.info_for_sort = []  #
#         self.rtsp_nums = None  # 当前为2
#
#     def init_list(self):
#         if self.rtsp_nums:
#             for i in range(self.rtsp_nums):
#                 self.info_for_sort[i] = None
'''
info_for_sort = [
                [[p1,conf1,cls1,img1],[p1,conf2,,cls2，img2]],#rtsp1 同一张图片存了多次
                [[p2,conf3,,img3，cls3],[p2,conf4,,img4，cls4]] #rtsp2
                ]
'''