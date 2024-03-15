from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy
from pycocotools import mask as coco_mask
from PIL import Image

def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, poly=False):
    start_anno_id = anno_id
    x, y, width, height = bounding_box_from_mask(image_mask)
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours = measure.find_contours(image_mask, 0.5)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
    }
    if poly == False:
        annotation["segmentation"] = encoded_mask
        annotation["segmentation"]["counts"] = str(
            annotation["segmentation"]["counts"], "utf-8"
        )
    if poly == True:
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            sc = bunch_coords(segmentation)
            sc = simplify_coords_vwp(sc, 2)
            sc = unbunch_coords(sc)
            annotation["segmentation"].append(sc)
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, img_size=[1280,720], categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder
        self.image_names = sorted(os.listdir(os.path.join(self.dataset_folder, "images")))
        self.image_names = [
            os.path.split(name)[1]
            for name in self.image_names
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") or name.endswith(".JPG")
        ]
        self.coco_json_path = coco_json_path
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.categories = [
            category["name"] for category in self.coco_json["categories"]
        ]
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        self.global_annotation_id = len(self.coco_json["annotations"])
        # self.category_colors = distinctipy.get_colors(len(self.categories))
        # self.category_colors = [
        #     tuple([int(255 * c) for c in color]) for color in self.category_colors
        # ]
        self.category_colors = [(0, 0, 255), (0, 255, 0), (0, 125, 255), (0, 255, 255), (255, 255, 0), (255, 125, 0),
                                (255, 0, 0), (255, 0, 125), (255, 0, 255)]

    def __init_coco_json(self, categories):
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]

    def get_last_imageid(self,annotations):
        last_anno=annotations[-1]
        image_id=last_anno['image_id']
        return image_id

    def get_last_annoid(self,annotations):
        last_anno=annotations[-1]
        id=last_anno['id']
        return id

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.coco_json["images"][image_id]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding,os.path.basename(image_name)

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def update_annotations(self, image_id, annotation_ids, new_label):
        # print("Update label to", new_label, "for", annotation_ids, "in dataexplorer")
        label_id = self.get_categories().index(new_label)

        for annotation_id in annotation_ids:
            for annotation in self.coco_json["annotations"]:
                if (
                    annotation["image_id"] == image_id
                    and annotation["id"] == annotation_id
                ):
                    annotation["category_id"] = label_id

        for annotation in self.annotations_by_image_id[image_id]:
            if annotation["id"] in annotation_ids:
                annotation["category_id"] = label_id

    def clear_annotations(self, image_id, annotation_ids):
        for annotation_id in annotation_ids:
            for annotation in self.coco_json["annotations"]:
                if (
                    annotation["image_id"] == image_id
                    and annotation["id"] == annotation_id
                ):  # and annotation["id"] in annotation_ids:
                    self.coco_json["annotations"].remove(annotation)

        # iterate over a copy of the list annotaiton_by_image_id[image_id]
        # because implace modification of the list causes discrapancies with the list index
        for annotation in self.annotations_by_image_id[image_id][:]:
            if annotation["id"] in annotation_ids:
                self.annotations_by_image_id[image_id].remove(annotation)

    def add_annotation(self, image_id, category_id, mask, poly=True):
        poly=False
        if mask is None:
            return
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, poly=poly
        )
        self.__add_to_our_annotation_dict(annotation)
        self.coco_json["annotations"].append(annotation)
        self.global_annotation_id += 1

    def save_annotation(self):
        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f)

    def save_buffer_mask(self,image_id):
        merged_mask = np.zeros([self.img_h, self.img_w])

        masks=[]
        for ann in self.coco_json["annotations"]:
            if ann['image_id']==image_id:
                masks.append(ann["segmentation"])
        for rle in masks:
            merged_mask += coco_mask.decode(rle)
        merged_mask[merged_mask > 0] = 1
        final_mask = np.zeros([self.img_h, self.img_w, 3])
        final_mask[merged_mask == 1] = [255, 0, 0]
        mask_img = Image.fromarray(np.uint8(final_mask), "RGB").convert("P")
        mask_img.save("./salt/mask.png")
