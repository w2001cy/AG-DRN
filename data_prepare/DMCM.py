import cv2 as cv
import numpy as np
import random
import os
import math
from filling import compensate
from cut import get_new_gt2, collect_edges

HEIGHT = 400
WIDTH = 400
final_lon = 256

def create_fish(srcImg):
    up = []
    down = []
    left = []
    right = []
    dstImg = np.zeros([HEIGHT, WIDTH, 3], np.uint8)
    det_uv = np.zeros([HEIGHT, WIDTH, 2], np.int32) + 500

    x0 = (WIDTH - 1)/ 2.
    y0 = (HEIGHT - 1) / 2.

    r = random.randint(0, int(WIDTH/2))

    x_r = (WIDTH - 1)/ 2. + r
    y_r = (HEIGHT - 1) / 2. + r


    min_x = 9999
    cut = 0
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            x_ = (i - x0)
            y_ = (j - y0)
            OO1 = (x_r ** 2 - x_ ** 2 - y_ ** 2) / (2 * abs(y_))
            r1 = math.sqrt((OO1 + abs(y_)) ** 2 + x_ ** 2)
            y_1 = r1 - OO1 if y_ > 0 else -(r1 - OO1)
            x_1 = x_

            OO2 = (y_r ** 2 - y_ ** 2 - x_ ** 2) / (2 * abs(x_))
            r2 = math.sqrt((OO2 + abs(x_)) ** 2 + y_ ** 2)
            x_2 = r2 - OO2 if x_ > 0 else -(r2 - OO2)
            y_2 = y_

            x = (x_1 + x_2) / 2
            y = (y_1 + y_2) / 2

            if (int(y) == int(-y0) and x >= -x0 and x <= x0):
                if (x < min_x):
                    min_x = x
                    cut = -x_
    cut_r = cut
    start = int(x0 - cut_r)
    end = int(x0 + cut_r) + 1

    for i in range(start, end):
        for j in range(start, end):
            x_ = (j - x0)
            y_ = (i - y0)
            OO1 = (x_r ** 2 - x_ ** 2 - y_ ** 2) / (2 * abs(y_))
            r1 = math.sqrt((OO1 + abs(y_)) ** 2 + x_ ** 2)
            y_1 = r1 - OO1 if y_ > 0 else -(r1 - OO1)
            x_1 = x_

            OO2 = (y_r ** 2 - y_ ** 2 - x_ ** 2) / (2 * abs(x_))
            r2 = math.sqrt((OO2 + abs(x_)) ** 2 + y_ ** 2)
            x_2 = r2 - OO2 if x_ > 0 else -(r2 - OO2)
            y_2 = y_

            x = (x_1 + x_2) / 2
            y = (y_1 + y_2) / 2

            u = int(round(x + x0))
            v = int(round(y + y0))

            if (u >= 0 and u < WIDTH) and (v >= 0 and v < HEIGHT):
                dstImg[i, j, 0] = srcImg[v, u, 0]
                dstImg[i, j, 1] = srcImg[v, u, 1]
                dstImg[i, j, 2] = srcImg[v, u, 2]

                up, down, left, right = collect_edges(start, end, j, i, u, v, up, down, left, right)
                cut_r_gt = int(x0 - up[0][0])
                parameter_c = float(cut_r_gt) / float(cut_r)
                parameter_b = float(final_lon) / float(cut_r_gt*2)
                det_uv[v, u, 0] = (((parameter_c * x_) + x0) - u) * parameter_b
                det_uv[v, u, 1] = (((parameter_c * y_) + y0) - v) * parameter_b

    cropImg = dstImg[(int(x0) - int(cut_r)):(int(x0) + int(cut_r)), (int(y0) - int(cut_r)):(int(y0) + int(cut_r))]
    dstImg2 = cv.resize(cropImg, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)

    det_u = det_uv[:, :, 0]
    det_v = det_uv[:, :, 1]

    det_u = compensate(det_u)
    det_v = compensate(det_v)


    source_Img, det_u, det_v = get_new_gt2(srcImg, det_u, det_v, up, down, left, right)
    source_Img = source_Img[(int(x0) - int(cut_r_gt)):(int(x0) + int(cut_r_gt)), (int(y0) - int(cut_r_gt)):(int(y0) + int(cut_r_gt))]
    source_Img = cv.resize(source_Img, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)

    return dstImg, dstImg2, source_Img


path = 'D:/fish/code/COCOdata/train2017/'
num = 1
mode = 'train'

if __name__ == "__main__":
    for root, dirs, img_list in os.walk(path):
        random.shuffle(img_list)
        print(img_list)

    os.makedirs(f'../dataset/data/{mode}', exist_ok=True)
    os.makedirs(f'../dataset/gt/{mode}', exist_ok=True)

    start_index = 0

    for files in img_list[start_index:]:

        print(files)
        srcImg = cv.imread('D:/fish/code/COCOdata/train2017/' + files)
        srcImg = cv.resize(srcImg, (400, 400))
        dstImg, cutImg, source_Img = create_fish(srcImg)

        cv.imwrite('../dataset/data/' + mode + '/' + str(num) + '.jpg', cutImg)
        cv.imwrite('../dataset/gt/' + mode + '/' + str(num) + '.jpg', source_Img)
        num = num + 1
        if num == 30001:
            break
