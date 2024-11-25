import os
import json
import cv2
import numpy as np
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from torchvision.ops import nms

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def clip_points_to_image(points, width, height):
    """裁剪点到图像边界内"""
    clipped_points = []
    for point in points:
        if isinstance(point, (list, np.ndarray)) and len(point) == 2:
            x, y = point
            x = min(max(0, x), width - 1)
            y = min(max(0, y), height - 1)
            clipped_points.append((x, y))
    return clipped_points


def process_contours(contours, width, height):
    """处理轮廓，去除超出图像边界的点，并合并重复的轮廓"""
    processed_contours = []
    for contour in contours:
        if contour.ndim == 2 and contour.shape[1] == 2:  # 确保是二维坐标
            contour = contour.squeeze().tolist()
            contour = clip_points_to_image(contour, width, height)
            processed_contours.append(contour)

    # 合并重复的轮廓
    unique_contours = []
    seen_contours = set()
    for contour in processed_contours:
        contour_tuple = tuple(map(tuple, contour))
        if contour_tuple not in seen_contours:
            seen_contours.add(contour_tuple)
            unique_contours.append(contour)

    return unique_contours


def predict_and_save_results(image_dir, output_dir, cfg, max_attempts=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictor = DefaultPredictor(cfg)

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            attempts = 0
            success = False

            while attempts < max_attempts and not success:
                attempts += 1
                try:
                    image = cv2.imread(image_path)
                    outputs = predictor(image)

                    instances = outputs["instances"].to("cpu")

                    # 获取预测的边界框、得分和类别
                    boxes = instances.pred_boxes.tensor
                    scores = instances.scores
                    labels = instances.pred_classes

                    # 手动执行 NMS
                    keep_indices = nms(boxes, scores, iou_threshold=0.4)  # 阈值可以根据需要调整

                    # 只保留 NMS 后的结果
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]
                    labels = labels[keep_indices]

                    result = {
                        "version": "1.0.0",
                        "flags": {},
                        "shapes": []
                    }

                    for i in range(len(keep_indices)):
                        category_id = int(labels[i].item())
                        label = list(LABEL_TO_ID.keys())[list(LABEL_TO_ID.values()).index(category_id)]

                        mask = instances.pred_masks[keep_indices[i]].numpy()
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            points = contour.squeeze().tolist()

                            # 如果点数不足4个，重复最后一个点来补足
                            while len(points) < 4:
                                points.append(points[-1])

                            # 确保每个点是二维坐标
                            if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in points):
                                points = [[round(coord, 4) for coord in point] for point in points]
                            else:
                                logging.error(f"Invalid points structure: {points}")
                                continue

                            result["shapes"].append({
                                "label": label,
                                "points": points,
                                "group_id": None,
                                "shape_type": "polygon",
                                "flags": {}
                            })

                    result.update({
                        "imagePath": image_name,
                        "imageData": None,
                        "imageHeight": image.shape[0],
                        "imageWidth": image.shape[1]
                    })

                    json_file_name = os.path.splitext(image_name)[0] + '_predictions.json'
                    json_file_path = os.path.join(output_dir, json_file_name)
                    with open(json_file_path, 'w') as json_file:
                        json.dump(result, json_file, indent=4)

                    success = True
                    logging.info(f"Prediction for {image_name} successful after {attempts} attempt(s).")

                except Exception as e:
                    logging.error(f"Error processing {image_name} on attempt {attempts}: {e}")

            if not success:
                logging.error(f"Failed to produce valid prediction for {image_name} after {max_attempts} attempts.")


def main():
    pre_image_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\detectron2-main\datasets\data\Validation-Public"
    save_output_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\detectron2-main\datasets\data\result\9.26"

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 30000
    cfg.OUTPUT_DIR = "./output"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predict_and_save_results(pre_image_dir, save_output_dir, cfg)

if __name__ == '__main__':
    main()
