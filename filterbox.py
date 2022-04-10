import cv2
import torch
import numpy as np
from seetings_globals import *
import utils.general
import datetime
# from utils.plots import colors, plot_one_box
from shapely.geometry import Polygon, MultiPoint
from utils.general import (
    rectlong2opencv, check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from shapely.geometry import Point
G = Global
K1 = 1.5
horizon = 0.3
vertic = 0.5
gbal_opt = None

def need_boxs(bbox, cls_ids):
    personlist = []
    person_clslist = []
    no_belt_list = []
    tizi_list = []
    for i in range(len(bbox)):
        if cls_ids[i] == 2 or cls_ids[i] == 3: 
            personlist.append(bbox[i])
            person_clslist.append(cls_ids[i])
        if cls_ids[i] == 0:  
            no_belt_list.append(bbox[i])
        if cls_ids[i] == 5:
            tizi_list.append(bbox[i])
    return personlist, person_clslist, no_belt_list, tizi_list


def xywha2xyxyxyxy(x, img, objtype):
    x = torch.tensor(x)
    maxbox = Polygon([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])])
    x = np.array(x)
    x[-1] *= 180 / np.pi
    opencv_rect = rectlong2opencv(x)
    cx, cy, w, h, angle = opencv_rect
    # new_angle = calcu_new_angle(angle)
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect).astype(int)
    poly_box = Polygon([(box[0][0], box[0][1]), (box[1][0], box[1][1]), (box[2][0], box[2][1]), (box[3][0], box[3][1])])


    # print(poly_box.intersection(maxbox).exterior.coords,'cccccccccccc')
    # for item in poly_box.intersection(maxbox).exterior.coords:
    #     print(item)
    #     _x, _y = item.xy
    # print(type(poly_box.intersection(maxbox).exterior.coords))
    # if isinstance(poly_box.intersection(maxbox).exterior.coords,list):
    #     print('got list error','eeeeeeeeeeeee')
    #     shapely.coords.CoordinateSequence()
    #     print(dir(poly_box.intersection(maxbox).exterior.coords))
    _x, _y = poly_box.intersection(maxbox).exterior.coords.xy

    if (len(_x) == 5):
        for k, (i, j) in enumerate(zip(list(_x), list(_y))):

            if k <= 3:
                box[k] = [i, j]
    else:
        data = np.concatenate((np.array(_x[:-1]).reshape(len(_x) - 1, -1), np.array(_y[:-1]).reshape(len(_x) - 1, -1)),
                              -1).astype(int)
        rect = cv2.minAreaRect(data)
        box = cv2.boxPoints(rect).astype(int)

    insu4cor = []
    person4cor = []
    if objtype == 4:
        # print('one box',box)
        for onepoint in box:
            px = onepoint[0]
            py = onepoint[1]
            # p = [px,py]
            insu4cor.append(px)
            insu4cor.append(py)
        # print('insu4corrrrrrrrrrrr',insu4cor)
        return box, insu4cor
    else:
        for onepoint in box:
            px = onepoint[0]
            py = onepoint[1]
            p = [px, py]
            person4cor.append(px)
            person4cor.append(py)
            # print(box,'1111', person4cor,'2222')
        return box, person4cor
import copy
import math
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
def plot_one_box2(img, insulatorlist,objtype,path):
    # print('plot_one_box2plot_one_box2')
    cpimg = copy.deepcopy(img)
    pilimg = Image.fromarray(cpimg.astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(pilimg)
    rects = []
    # print(insulatorlist)
    # temp = [[804.343750 ,344.500000 ,33.000000 ,123.000000, -90],
    #         [443.404727, 320.153665 ,38.887493 ,146.225353, -90],
    #         [772.843750 ,325.500000 ,38.000000, 139.000000 ,-90],
    #         [1096.343750 ,320.000000 ,37.000000 ,142.000000 ,-90],
    #         [515.343750, 341.500000, 35.000000 ,133.000000, -90]  ]
    # if objtype == 'insu':
    #     insulatorlist = temp
    # else:
    #     insulatorlist = insulatorlist
    for xx in insulatorlist:
        # print(xx)
        x = np.array(xx)
        x[-1] *= 180 / np.pi
        opencv_rect = rectlong2opencv(x)

        cx, cy, w, h, angle = opencv_rect
        # if angle
        # print(angle)

        if -24 <= angle <= -20:
            angle -= 2
            print('-30 minus some')
        if -67<= angle <= -65:
            angle += 10
            print('30 add some')
        if -90 <= angle < 0:
            # print('input angle is ',angle)
            angle = 90 - angle
        # print('angle after 90 - :',angle)
        '''responce to -30degrees, negtive too much ,add some'''



        anglePi = -(angle) * math.pi / 180.0
        # rect = ((cx, cy), (w, h), 90 + angle)
        x = cx
        y = cy
        width = w
        height = h

        '''force head height=width for show'''
        if objtype == 'head':
            width = height = max(width,height)

        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)

        x1 = x - 0.5 * width
        y1 = y - 0.5 * height

        x0 = x + 0.5 * width
        y0 = y1

        x2 = x1
        y2 = y + 0.5 * height

        x3 = x0
        y3 = y2

        x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
        y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

        x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
        y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

        x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
        y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

        x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
        y3n = (x3 - x) * sinA + (y3 - y) * cosA + y
        draw.line([(x0n, y0n), (x1n, y1n)], fill=(0, 255,0),width=2)
        draw.line([(x1n, y1n), (x2n, y2n)], fill=(0, 255,0),width=2)
        draw.line([(x2n, y2n), (x3n, y3n)], fill=(0, 255,0),width=2)
        draw.line([(x0n, y0n), (x3n, y3n)], fill=(0, 255,0),width=2)
        '''ind shorter length'''
        if objtype =='insu':
            pix_width = str(min(width,height)) #insu use short side
        else:
            pix_width = str(max(width, height)) #head use long side

        font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
        # draw.text((x0n, y0n),min_width,fill=(0, 0, 255), font=font)
        # plt.imshow(pilimg)
        # plt.show()
        box = [x0n, y0n,x1n, y1n,x2n, y2n,x3n, y3n]
        rects.append(box)

        # break
    cpimg = np.array(pilimg)
    # file_name = str(path).split('/')[-1]
    # print(path)
    # path = os.path.join('./inference/paper/',file_name)
    # cv2.imwrite(path,cpimg)
    return rects,cpimg

def add_dis2cam_onbox(obj_point,label,pix_width,img):
    # print(obj_point,'pointttttttttttt')
    cpimg = copy.deepcopy(img)
    pilimg = Image.fromarray(cpimg.astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
    # draw.text((obj_point[0], obj_point[1]), str(label)[:3]+str(pix_width)[:3]+'pix',fill=(0, 0, 255), font=font)
    draw.text((obj_point[0], obj_point[1]), str(label)[:3] , fill=(0, 0, 255), font=font)
    # plt.imshow(pilimg)
    # plt.show()
    deepimg2 = np.array(pilimg)
    return deepimg2
    pass

def getrepresent_point(oenrect, objtype):
    allx = [oenrect[0], oenrect[2], oenrect[4], oenrect[6]]
    allx.sort()
    ally = [oenrect[1], oenrect[3], oenrect[5], oenrect[7]]
    ally.sort()
    minx = allx[0]
    miny = ally[0]
    maxx = allx[3]
    maxy = ally[3]
    leftuppoint = (minx, miny)
    rightdownpoint = (maxx, maxy)
    waijie2point = [leftuppoint, rightdownpoint]  

    def take_y(elem):
        return elem[1]

    p1 = (oenrect[0], oenrect[1])
    p2 = (oenrect[2], oenrect[3])
    p3 = (oenrect[4], oenrect[5])
    p4 = (oenrect[6], oenrect[7])
    four_points = [p1, p2, p3, p4]
    four_points.sort(reverse=True, key=take_y) 
    lowest_2points = four_points[:2]
    x1 = lowest_2points[0][0]
    y1 = lowest_2points[0][1]
    x2 = lowest_2points[1][0]
    y2 = lowest_2points[1][1]
    if objtype == 'insulator': 
        Insumidpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        InsuWidth = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 
        return Insumidpoint, InsuWidth, waijie2point
    else:
        highest_2points = four_points[2:4]  
        x1 = highest_2points[0][0]
        y1 = highest_2points[0][1]
        x2 = highest_2points[1][0]
        y2 = highest_2points[1][1]
        HeadUpmidpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        HeadWidth = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 
        return HeadUpmidpoint, HeadWidth, waijie2point

def process_height_result(img, bboxs, cls_confs, cls_idss, path ,opt,vid_writer):
    insulatorlist = cls_conf = personlist = []
    insuconflist = [0 for x in range(0, 20)]
    if opt.simulate:
        print('warning!!! simulating!!!')
        for bbox, cls_ids, cls_conf in zip(bboxs, cls_idss, cls_confs):  
            insuconflist = [cls_conf[i] for i in range(len(bbox)) if cls_ids[i] == 1]
            personlist = [bbox[i] for i in range(len(bbox)) if cls_ids[i] == 0]  #
    else:
        for bbox, cls_ids, cls_conf in zip(bboxs, cls_idss, cls_confs): 
            insulatorlist = [bbox[i] for i in range(len(bbox)) if cls_ids[i] == 4] 

            insuconflist = [cls_conf[i] for i in range(len(bbox)) if cls_ids[i] == 4]
            personlist = [bbox[i] for i in range(len(bbox)) if cls_ids[i] == 3 or cls_ids[i] == 3] 
            # personlist = [bbox[i] for i in range(len(bbox)) if cls_ids[i] == 3]  #



    show_original_labels = True
    insuInfo = []
    headInfo = []
    # for i, item in enumerate(insulatorlist):
    insu_rects,deep_img = plot_one_box2(img, insulatorlist,objtype='insu',path=path)
    head_rects, deep_img = plot_one_box2(deep_img, personlist, objtype='head',path=path)

    for rect in insu_rects:
        Insumidpoint, InsuWidth, waijie2point = getrepresent_point(rect, objtype='insulator')
        insuInfo.append([Insumidpoint, InsuWidth, '0', rect, waijie2point,rect])
    #head
    for rect in head_rects:
        HeadUpmidpoint, HeadWidth, waijie2point = getrepresent_point(rect, objtype='Head')
        #print(rect)
        headInfo.append([HeadUpmidpoint, HeadWidth, '0', rect, waijie2point,rect])



    objNperList = []
    if len(insuInfo) > 0 or len(headInfo) > 0:
        # print('L NOT EMPTY,CALCULATE DISTANCE.')
        for ee, i in enumerate(headInfo):  # 全连接对 std_per_cor_list
            for e, j in enumerate(insuInfo):
                NOid_item = [i, j]  # [[Hpoint,Hwidth],[InsuPoint,InsuWidth]]
                objNperList.append(NOid_item)  # cupNperList = [(p1,c1,objID),(p1,c2,objID),...]  注意顺序！！！

        for one_pair in objNperList:
            '''[HeadUpmidpoint, HeadWidth, '0', rect, waijie2point]'''
            headInfo = one_pair[0]
            insuInfo = one_pair[1]
            head_pixel_width, head_dis = get_dis_to_cams(headInfo, Objtype='head')
            insu_pixel_width, insu_dis = get_dis_to_cams(insuInfo, Objtype='insu')

            # deep_img = add_dis2cam_onbox(headInfo[0],head_dis,head_pixel_width,deep_img) # pass in midpoint only
            #deep_img = add_dis2cam_onbox(insuInfo[0], insu_dis,insu_pixel_width,deep_img)


            vertical_dis_diff = abs(head_dis - insu_dis)
            mean_D,horizontal_dis_diff = get_horizontal_len(headInfo, insuInfo, head_dis, insu_dis)  # 水平距离
            # plot_one_box(headInfo[2], img, label=str(head_dis)[:3], color=(0, 0, 255))
            # plot_one_box(insuInfo[2], img, label=str(insu_dis)[:3], color=(100, 255, 100))
            this_pair_obj_dis_plane = (vertical_dis_diff ** 2 + horizontal_dis_diff ** 2) ** 0.5
            head_represent_point = headInfo[0] #头部的代表点
            insu_represent_point = insuInfo[0]#绝缘子的代表点
            y_diff_pix = abs(insu_represent_point[1] - head_represent_point[1])

            '''S0 on projected plane'''
            real_ydiff = get_vertical_height_diff(y_diff_pix, mean_D)
            '''S in space, final distance'''
            real_dis = (this_pair_obj_dis_plane ** 2 + real_ydiff ** 2) ** 0.5
            if real_dis <= 3:
                #print(headInfo[-1])
                #print(insuInfo[-1])
                x0n = headInfo[-1][0]
                y0n = headInfo[-1][1]
                x1n = headInfo[-1][2]
                y1n = headInfo[-1][3]
                x2n = headInfo[-1][4]
                y2n = headInfo[-1][5]
                x3n = headInfo[-1][6]
                y3n = headInfo[-1][7]

                a0n = insuInfo[-1][0]
                b0n = insuInfo[-1][1]
                a1n = insuInfo[-1][2]
                b1n = insuInfo[-1][3]
                a2n = insuInfo[-1][4]
                b2n = insuInfo[-1][5]
                a3n = insuInfo[-1][6]
                b3n = insuInfo[-1][7]




                cpimg = copy.deepcopy(deep_img)
                pilimg = Image.fromarray(cpimg.astype('uint8')).convert('RGB')
                draw = ImageDraw.Draw(pilimg)
                #head_draw_RED
                draw.line([(x0n, y0n), (x1n, y1n)], fill=(0, 0, 255), width=2)
                draw.line([(x1n, y1n), (x2n, y2n)], fill=(0, 0, 255), width=2)
                draw.line([(x2n, y2n), (x3n, y3n)], fill=(0, 0, 255), width=2)
                draw.line([(x0n, y0n), (x3n, y3n)], fill=(0, 0, 255), width=2)
                # jueyuanzi_draw_RED
                draw.line([(a0n, b0n), (a1n, b1n)], fill=(0, 0, 255), width=2)
                draw.line([(a1n, b1n), (a2n, b2n)], fill=(0, 0, 255), width=2)
                draw.line([(a2n, b2n), (a3n, b3n)], fill=(0, 0, 255), width=2)
                draw.line([(a0n, b0n), (a3n, b3n)], fill=(0, 0, 255), width=2)

                #plt.imshow(pilimg)
                #plt.show()
                deep_img = np.array(pilimg)

                draw_one_pair_line(headInfo[0], insuInfo[0], real_dis, deep_img,head_pixel_width,insu_pixel_width)

            # file_name = str(path).split('/')[-1]
            # gpath = os.path.join('./inference/yanshi/results', file_name)

        cv2.imshow('deep', deep_img)
        cv2.waitKey(10)
        return deep_img


    #
    # ##return deep_img
    #     # vid_writer.write(deep_img)


def draw_one_pair_line(headpoint, insupoint, this_pair_obj_dis, im,head_pixel_width,insu_pixel_width):
    # print('draw')(BGR)
    cv2.line(im, headpoint, insupoint, (0, 0, 255), thickness=2, lineType=8)

    line_middle = (int((headpoint[0] + insupoint[0]) / 2), int((headpoint[1] + insupoint[1]) / 2))  # (0, 255, 0) green
    # cv2.putText(im, str(this_pair_obj_dis)[:3]+'m'+str(vertical_dis_diff)[:3], line_middle, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(im, str(this_pair_obj_dis)[:4] , line_middle, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # cv2.putText(im, str(this_pair_obj_dis)[:4] , insupoint, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)



# global_F = 1943.18
global_F = 2272.7
global_F = 280

real_insu_width = 0.35
def get_dis_to_cams(objInfo, Objtype):
    F = global_F
    # '''m WEIGHTS  not good'''
    # F = 1772.7  # according to 2-8 m_new_weights_file w = 0.25

    if Objtype == 'head':
        real_width = 0.26  # human head:25cm
    else:
        real_width = real_insu_width


    pixel_width = objInfo[1]
    # print('pix width:', objInfo[1])
    dis = F * real_width / pixel_width  # dis to camera
    dis_label = str(F * real_width / pixel_width) + 'm_' + str(pixel_width)[:4] #dis to camera and pixwidth
    return str(pixel_width), dis


def get_horizontal_len(headInfo, insuInfo, head_dis, insu_dis):
    F = global_F
    '''on projected plane: distance of W is mean of d1 and d2'''
    head_centerX = headInfo[0][0]
    insu_centerX = insuInfo[0][0]
    D = (head_dis+insu_dis)/2
    pix_horizontal_diff = abs(head_centerX - insu_centerX)
    W = D * pix_horizontal_diff / F
    return D,W  # W is real value of W, D is real mean dis

def get_vertical_height_diff(pix_height, mean_D):
    F = global_F
    H = mean_D * pix_height / F
    return H
