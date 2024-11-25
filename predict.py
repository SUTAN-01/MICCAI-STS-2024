import os
import json
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 定义字符标签到 ID 的映射
LABEL_TO_ID = {
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4, "16": 5, "17": 6, "18": 7,
    "21": 8, "22": 9, "23": 10, "24": 11, "25": 12, "26": 13, "27": 14, "28": 15,
    "31": 16, "32": 17, "33": 18, "34": 19, "35": 20, "36": 21, "37": 22, "38": 23,
    "41": 24, "42": 25, "43": 26, "44": 27, "45": 28, "46": 29, "47": 30, "48": 31,
    "51": 32, "52": 33, "53": 34, "54": 35, "55": 36,
    "61": 37, "62": 38, "63": 39, "64": 40, "65": 41,
    "71": 42, "72": 43, "73": 44, "74": 45, "75": 46,
    "81": 47, "82": 48, "83": 49, "84": 50, "85": 51
}

# 定义所有标签的列表
CLASS_NAMES = list(LABEL_TO_ID.keys())

def predict_and_save_results(image_dir, output_dir, cfg):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用训练完成的模型进行预测
    predictor = DefaultPredictor(cfg)

    # 获取元数据
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            outputs = predictor(image)

            instances = outputs["instances"].to("cpu")
            result = {
                "version": "1.0.0",
                "flags": {},
                "shapes": []
            }

            for i in range(len(instances)):
                instance = instances[i]

                # 确保 `instance.pred_classes` 和 `instance.pred_masks` 有效
                if len(instance.pred_classes) == 0 or len(instance.pred_masks) == 0:
                    print(f"Warning: No predictions found for image {image_name}. Skipping.")
                    continue

                category_id = int(instance.pred_classes.item())  # 只需获取当前实例的类别
                label = list(LABEL_TO_ID.keys())[list(LABEL_TO_ID.values()).index(category_id)]

                mask = instance.pred_masks.numpy()[0]  # 只需获取当前实例的mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    points = contour.squeeze().tolist()

                    # 确保点的格式为整数对
                    if isinstance(points[0], list):  # 检查是否有多个点
                        points = [[int(point[0]), int(point[1])] for point in points]
                    else:  # 如果只有一个点，确保它是整数对
                        points = [int(points[0]), int(points[1])]

                    result["shapes"].append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

            # 固定 imageHeight 和 imageWidth 的值
            result.update({
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": image.shape[0],  # 从原始图像中获取高度
                "imageWidth": image.shape[1]    # 从原始图像中获取宽度
            })

            json_file_name = os.path.splitext(image_name)[0] + '_predictions.json'
            json_file_path = os.path.join(output_dir, json_file_name)
            with open(json_file_path, 'w') as json_file:
                json.dump(result, json_file, indent=4)

            # 可视化检测结果
            v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
            output_image = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

            # 保存可视化结果
            vis_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '_vis.jpg')
            cv2.imwrite(vis_image_path, output_image)

if __name__ == '__main__':
    # 预测图路径和json保存地址
    pre_image_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\detectron2-main\datasets\data\Validation-Public"
    save_output_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\detectron2-main\datasets\data\result\9.26"

    # 加载配置
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(r"configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\detectron2-main\output\model_final.pth"  # 指定已训练的模型权重路径

    # 预测并保存结果
    predict_and_save_results(pre_image_dir, save_output_dir, cfg)
