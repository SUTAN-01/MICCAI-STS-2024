import time
import random
import cv2
import os
import numpy as np
from skimage.util import random_noise
import json
from copy import deepcopy

class DataAugmentForObjectDetection:
    def __init__(self, flip_rate=0.5, is_filp_pic_bboxes=True):
        # 只保留水平翻转的初始化参数
        self.flip_rate = flip_rate
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    def _flip_pic_bboxes(self, img, json_info):
        h, w, _ = img.shape
        sed = random.random()

        # 仅水平翻转
        if 0 < sed < 0.5:
            flip_img = cv2.flip(img, 1)  # 水平翻转
            inver = 1
        else:
            flip_img = img  # 不进行翻转
            inver = -1

        # 更新掩码中的坐标信息
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                if inver == 1:  # 只在水平翻转的情况下进行处理
                    p[0] = w - p[0]  # 更新 x 坐标为 w - x

        return flip_img, json_info

    def dataAugment(self, img, dic_info):
        if self.is_filp_pic_bboxes:
            if random.random() < self.flip_rate:
                img, dic_info = self._flip_pic_bboxes(img, dic_info)

        return img, dic_info


class ToolHelper:
    def parse_json(self, path):
        with open(path) as f:
            json_data = json.load(f)
        return json_data

    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)

    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)

if __name__ == '__main__':
    need_aug_num = 3 # 每张图片需要增强的次数

    toolhelper = ToolHelper()  # 工具
    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    source_img_path = r'D:\Users\25427\Desktop\bsxm\detectron2-main\datasets\data\daliat_train_label\Images'
    json_path_dir = r'D:\Users\25427\Desktop\bsxm\detectron2-main\datasets\data\daliat_train_label\Masks'
    save_img_json_path = r'D:\Users\25427\Desktop\bsxm\detectron2-main\datasets\data\daliat_train_label\augmented_imgs'
    save_json_path = r'D:\Users\25427\Desktop\bsxm\detectron2-main\datasets\data\daliat_train_label\augmented_labels'

    if not os.path.exists(save_img_json_path):
        os.makedirs(save_img_json_path)
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)

    image_files = [f for f in os.listdir(source_img_path) if f.endswith(('jpg', 'png'))]

    total_images = len(image_files) * need_aug_num
    cur_image_num = 0

    for image_file in image_files:
        cur_image_num += 1
        print(f'Processing image {cur_image_num}/{total_images} ...')
        img_path = os.path.join(source_img_path, image_file)
        json_path = os.path.join(json_path_dir, image_file.replace('.jpg', '_Mask.json'))

        img = cv2.imread(img_path)
        dic_info = toolhelper.parse_json(json_path)

        # 保存原始图像文件名
        original_image_file_name = image_file

        for i in range(need_aug_num):
            aug_img, aug_dic_info = dataAug.dataAugment(img, dic_info)

            save_img_name = f'{i}_{image_file}'
            save_json_name = f'{i}_{image_file.replace(".jpg", ".json")}'

            # 保存增强后的图像
            toolhelper.save_img(os.path.join(save_img_json_path, save_img_name), aug_img)

            # 保持原始图像文件名不变
            aug_dic_info['imagePath'] = original_image_file_name
            aug_dic_info['imageData'] = None  # 设置 imageData 为 None

            # 保存增强后的 JSON 文件
            toolhelper.save_json(save_json_name, save_json_path, aug_dic_info)

    print('Data augmentation complete!')
