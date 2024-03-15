import os, copy
import numpy as np
from salt.onnx_model import OnnxModels
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils

selected_annotations = []


class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if len(self.input_point) == 0:
            self.input_point = np.array([input_point])
        else:
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(
        self, onnx_models_path, dataset_path, img_size=[1280,720],categories=None, coco_json_path=None
    ):
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path,img_size=img_size, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories, self.category_colors = self.dataset_explorer.get_categories(
            get_colors=True
        )
        self.image_id = 0
        self.category_id = 0
        self.propogated_step = 0
        self.show_other_anns = True
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)

        self.full_img_path = None
        if self.image_name.endswith(".jpeg"):
            self.extension = self.image_name[-5:]
            self.image_name=self.image_name[:-5]

        else:
            self.extension = self.image_name[-4:]
            self.image_name = self.image_name[:-4]
        self.display = self.image_bgr.copy()
        self.onnx_helper = OnnxModels(
            onnx_models_path,
            image_width=self.image.shape[1],
            image_height=self.image.shape[0],
            name = self.image_name,
        )
        self.du = DisplayUtils()
        self.reset()

    def list_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        return anns, colors

    def get_last_imageid(self):
        return self.dataset_explorer.get_last_imageid(self.dataset_explorer.coco_json["annotations"])

    def get_last_annoid(self):
        return self.dataset_explorer.get_last_annoid(self.dataset_explorer.coco_json["annotations"])

    def update_img_name(self):
        self.full_img_path = os.path.join(self.dataset_path, "images", self.image_name+self.extension)


    def update_annotation_label(self, new_label, annotation_ids):
        self.dataset_explorer.update_annotations(self.image_id, annotation_ids, new_label)

    def clear_annotations(self, annotation_ids):
        self.dataset_explorer.clear_annotations(self.image_id, annotation_ids)

    def __draw_known_annotations(self, selected_annotations=[]):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        for i, (ann, color) in enumerate(zip(anns, colors)):
            for selected_ann in selected_annotations:
                if ann["id"] == selected_ann:
                    colors[i] = (0, 0, 255)
        # Use this to list the annotations
        self.display = self.du.draw_annotations(self.display, anns, colors)

    def __draw(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.draw_points(
                self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
            )
            self.display = self.du.overlay_mask_on_image(
                self.display, self.curr_inputs.curr_mask
            )
        if self.show_other_anns:
            self.__draw_known_annotations(selected_annotations)

    def add_click(self, new_pt, new_label, selected_annotations=[]):
        self.curr_inputs.add_input_click(new_pt, new_label)
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)
        self.__draw(selected_annotations)

    def remove_click(self, new_pt):
        print("ran remove click")
        remove_mask_ids = self.get_pt_mask(new_pt)
        self.clear_annotations(remove_mask_ids)
        # self.__draw(selected_annotations)

    def choose_annotation(self, point):
        return self.get_pt_mask(point)

    def reset(self, hard=True, selected_annotations=[]):
        self.curr_inputs.reset_inputs()
        self.__draw(selected_annotations)

    def toggle(self, selected_annotations=[]):
        self.show_other_anns = not self.show_other_anns
        self.__draw(selected_annotations)

    def get_pt_mask(self, pt):
        masks = []
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        image = self.display
        masks_ids = self.du.masks_containing_pts(pt, image, anns, colors)
        return masks_ids
        # print(mask[pt])

    def hover(self, pt=[], selected_annotations=[]):
        self.du.hover_mask_id = self.get_pt_mask(pt)
        self.display = self.image_bgr.copy()
        self.__draw_known_annotations(selected_annotations)
        self.du.hover_mask_id = []

    def step_up_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.increase_transparency()
        self.__draw(selected_annotations)

    def step_down_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.decrease_transparency()
        self.__draw(selected_annotations)

    def draw_selected_annotations(self, selected_annotations=[]):
        self.__draw(selected_annotations)

    def save_ann(self):
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask
        )

    def save(self):
        self.dataset_explorer.save_annotation()

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id += 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        else:
            self.image_name = self.image_name[:-4]
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        self.reset()

    def set_image_by_imageid(self,image_id,anno_id):
        if image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id = image_id
        self.dataset_explorer.global_annotation_id=anno_id+1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        else:
            self.image_name = self.image_name[:-4]
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        self.reset()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
            self.image_name,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        if self.image_name.endswith(".jpeg"):
            self.image_name=self.image_name[:-5]
        else:
            self.image_name = self.image_name[:-4]
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0],name = self.image_name)
        self.reset()

    def save_annotation_frommask(self,image_id,category_id,mask):
        self.dataset_explorer.add_annotation(image_id,category_id,mask)

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def save_mask(self):
        self.dataset_explorer.save_buffer_mask(self.image_id)

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name)
        self.category_id = category_id
