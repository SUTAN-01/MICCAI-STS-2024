from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog  # 确保导入了 MetadataCatalog
import cv2

# 配置模型
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
cfg.MODEL.WEIGHTS = r"D:\Users\25427\Desktop\bsxm\detectron2-main\output\model_final.pth"


# 创建预测器
predictor = DefaultPredictor(cfg)

# 打开一个测试图像
image = cv2.imread(r"D:\Users\25427\Desktop\bsxm\detectron2-main\datasets\data\Validation-Public\STS24_Train_Validation_000012.jpg")

# 获取模型的输出
outputs = predictor(image)

# 使用 Visualizer 可视化
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])  # 确保元数据存在
v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# 展示结果
cv2.imshow("Detection", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
