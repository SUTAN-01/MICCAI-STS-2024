import os
import json
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
from detectron2 import model_zoo

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


def load_json_annotations(json_file, image_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    instances = []

    for shape in data.get('shapes', []):
        label = shape.get('label')
        points = shape.get('points')

        if label is None or points is None:
            print(f"Warning: 'label' or 'points' not found in shape. Skipping shape: {shape}")
            continue

        category_id = LABEL_TO_ID.get(label)
        if category_id is None:
            print(f"Warning: Label '{label}' not found in LABEL_TO_ID. Skipping shape: {shape}")
            continue

        # Ensure points is a list of [x, y] pairs
        if not isinstance(points, list) or not all(isinstance(p, list) and len(p) == 2 for p in points):
            print(f"Warning: Invalid points format in shape. Skipping shape: {shape}")
            continue

        # Convert points to a flattened list of coordinates
        polygon = np.array(points, dtype=np.float32).flatten().tolist()

        # Compute bounding box from polygon points
        if len(points) > 0:
            points_np = np.array(points, dtype=np.float32)
            x_min, y_min = points_np.min(axis=0)
            x_max, y_max = points_np.max(axis=0)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            bbox = [0, 0, 0, 0]

        instance = {
            "segmentation": [polygon],
            "bbox": bbox,  # Set bbox as a list of [x, y, width, height]
            "bbox_mode": BoxMode.XYWH_ABS,  # This should be an integer or enum value
            "category_id": category_id
        }
        instances.append(instance)

    # Check if there is at least one valid annotation
    if not instances:
        print(f"Warning: No valid annotations found in {json_file}. Skipping this file.")
        return None

    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Cannot read image at {image_file}. Skipping this file.")
        return None

    return {
        "file_name": image_file,
        "image_id": os.path.basename(json_file).split('.')[0],
        "height": image.shape[0],
        "width": image.shape[1],
        "annotations": instances
    }


def register_dataset(dataset_name, json_dir, image_dir):
    def load_dataset():
        dataset_dicts = []
        for json_file in os.listdir(json_dir):
            if json_file.endswith('_Mask.json'):
                json_path = os.path.join(json_dir, json_file)
                image_file = os.path.join(image_dir, json_file.replace('_Mask.json', '.jpg'))
                data = load_json_annotations(json_path, image_file)
                if data is not None:
                    dataset_dicts.append(data)
        return dataset_dicts

    DatasetCatalog.register(dataset_name, load_dataset)
    MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES)


def reset_model_head(cfg):
    # 构建模型
    model = build_model(cfg)

    # 加载预训练模型的权重
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # 重新初始化头部层
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        model.roi_heads.box_predictor.cls_score.in_features,
        cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1,  # +1 for the background class
    )

    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        model.roi_heads.box_predictor.bbox_pred.in_features,
        cfg.MODEL.ROI_HEADS.NUM_CLASSES * 4,  # 4 for bbox coordinates
    )

    # Mask头权重（如果使用了Mask R-CNN）
    if hasattr(model.roi_heads, "mask_head"):
        model.roi_heads.mask_head.predictor = torch.nn.Conv2d(
            model.roi_heads.mask_head.predictor.in_channels,
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    return model
def predict_and_save_results(image_dir, output_dir, cfg):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用训练完成的模型进行预测
    predictor = DefaultPredictor(cfg)

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

                # 确保 instance.pred_classes 和 instance.pred_masks 有效
                if len(instance.pred_classes) == 0 or len(instance.pred_masks) == 0:
                    print(f"Warning: No predictions found for image {image_name}. Skipping.")
                    continue

                category_id = int(instance.pred_classes.item())  # 只需获取当前实例的类别
                label = list(LABEL_TO_ID.keys())[list(LABEL_TO_ID.values()).index(category_id)]

                mask = instance.pred_masks.numpy()[0]  # 只需获取当前实例的mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    points = contour.squeeze().tolist()

                    # 对每个点的坐标进行四舍五入，保留最多四位小数
                    #points = [[round(coord, 4) for coord in point] for point in points]

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


if __name__ == '__main__':
    # 预测图路径和json保存地址
    pre_image_dir = "./datasets/data/Validation-Public"
    save_output_dir = "./datasets/data/result"
    # 设置训练路径
    json_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\Masks"
    image_dir = r"D:\Users\25427\Desktop\kqsf\detectron2-main0\Images"
    register_dataset("my_dataset", json_dir, image_dir)
    cfg = get_cfg()
    print(cfg)  # Print the entire config to see its structure
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER =30000  # 训练迭代次数
    cfg.OUTPUT_DIR = "./output"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
############注释以下内容可以直接预测结果######
    # 训练模型
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # 保存训练后的模型权重
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    DetectionCheckpointer(trainer.model).save(best_model_path)
############################################