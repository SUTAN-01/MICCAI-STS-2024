# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""


# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "1"},
   {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "2"},
   {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "4"},
   {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "5"},
   {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "6"},
   {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "7"},
   {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "8"},
   {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "9"},
   {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "10"},
   {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "11"},
   {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "12"},
   {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "13"},
   {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "15"},
   {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "16"},
   {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "17"},
   {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "19"},
   {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "20"},
   {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "21"},
   {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "22"},
   {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "23"},
   {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "24"},
   {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "25"},
   {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "26"},
   {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "27"},
   {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "28"},
   {"color": [72, 0, 118], "isthing": 1, "id": 26, "name": "30"},
   {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "32"},
   {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "18"},
   {"color": [209, 0, 151], "isthing": 1, "id": 29, "name": "29"},
   {"color": [188, 208, 182], "isthing": 1, "id": 30, "name": "3"},
   {"color": [0, 220, 176], "isthing": 1, "id": 31, "name": "14"},
   {"color": [255, 99, 164], "isthing": 1, "id": 32, "name": "31"},
]
# COCO_CATEGORIES = [
#     {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "55"},
#     {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "54"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "53"},
#     {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "52"},
#     {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "61"},
#     {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "62"},
#     {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "63"},
#     {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "64"},
#     {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "65"},
#     {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "75"},
#     {"color": [230, 150, 140], "isthing": 1, "id": 11, "name": "74"},
#     {"color": [128, 64, 128], "isthing": 1, "id": 12, "name": "73"},
#     {"color": [244, 35, 232], "isthing": 1, "id": 13, "name": "72"},
#     {"color": [70, 70, 70], "isthing": 1, "id": 14, "name": "81"},
#     {"color": [102, 102, 156], "isthing": 1, "id": 15, "name": "82"},
#     {"color": [190, 153, 153], "isthing": 1, "id": 16, "name": "83"},
#     {"color": [180, 165, 180], "isthing": 1, "id": 17, "name": "84"},
#     {"color": [150, 100, 100], "isthing": 1, "id": 18, "name": "85"},
#     {"color": [107, 142, 35], "isthing": 1, "id": 19, "name": "11"},
#     {"color": [152, 251, 152], "isthing": 1, "id": 20, "name": "12"},
#     {"color": [70, 130, 180], "isthing": 1, "id": 21, "name": "13"},
#     {"color": [220, 20, 60], "isthing": 1, "id": 22, "name": "14"},
#     {"color": [230, 190, 255], "isthing": 1, "id": 23, "name": "15"},
#     {"color": [255, 0, 0], "isthing": 1, "id": 24, "name": "16"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 25, "name": "17"},
#     {"color": [255, 204, 54], "isthing": 1, "id": 26, "name": "48"},
#     {"color": [0, 153, 153], "isthing": 1, "id": 27, "name": "46"},
#     {"color": [220, 220, 0], "isthing": 1, "id": 28, "name": "45"},
#     {"color": [107, 142, 35], "isthing": 1, "id": 29, "name": "44"},
#     {"color": [152, 251, 152], "isthing": 1, "id": 30, "name": "43"},
#     {"color": [70, 130, 180], "isthing": 1, "id": 31, "name": "42"},
#     {"color": [220, 20, 60], "isthing": 1, "id": 32, "name": "41"},
#     {"color": [230, 190, 255], "isthing": 1, "id": 33, "name": "31"},
#     {"color": [255, 0, 0], "isthing": 1, "id": 34, "name": "32"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 35, "name": "33"},
#     {"color": [255, 204, 54], "isthing": 1, "id": 36, "name": "34"},
#     {"color": [0, 153, 153], "isthing": 1, "id": 37, "name": "35"},
#     {"color": [220, 220, 0], "isthing": 1, "id": 38, "name": "36"},
#     {"color": [119, 11, 32], "isthing": 1, "id": 39, "name": "37"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 40, "name": "38"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 41, "name": "21"},
#     {"color": [119, 11, 32], "isthing": 1, "id": 42, "name": "22"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 43, "name": "23"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 44, "name": "24"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 45, "name": "25"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 46, "name": "26"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 47, "name": "27"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 48, "name": "28"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 49, "name": "47"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 50, "name": "18"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 51, "name": "51"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 52, "name": "71"}
# ]
# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

# All Cityscapes categories, together with their nice-looking visualization colors
# It's from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  # noqa
CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]

# fmt: off
ADE20K_SEM_SEG_CATEGORIES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road, route", "bed", "window ", "grass", "cabinet", "sidewalk, pavement", "person", "earth, ground", "door", "table", "mountain, mount", "plant", "curtain", "chair", "car", "water", "painting, picture", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "tub", "rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove", "palm, palm tree", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus", "towel", "light", "truck", "tower", "chandelier", "awning, sunshade, sunblind", "street lamp", "booth", "tv", "plane", "dirt track", "clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "pool", "stool", "barrel, cask", "basket, handbasket", "falls", "tent", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light", "tray", "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass, drinking glass", "clock", "flag", # noqa
]
# After processed by `prepare_ade20k_sem_seg.py`, id 255 means ignore
# fmt: on


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 32, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    # assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_panoptic_standard":
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in COCO_CATEGORIES]
        thing_colors = [k["color"] for k in COCO_CATEGORIES]
        stuff_classes = [k["name"] for k in COCO_CATEGORIES]
        stuff_colors = [k["color"] for k in COCO_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(COCO_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))