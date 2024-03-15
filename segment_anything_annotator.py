import os
import argparse
import sys
import torch
from PyQt5.QtWidgets import QApplication
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization
from salt.editor import Editor
from salt.interface import ApplicationInterface
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from pycocotools import mask as mask_util

def resize_mask(mask,size):
    # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
    h, w = mask.shape[-2:]
    min_hw = min(h, w)
    return F.interpolate(mask, (int(h / min_hw * size), int(w / min_hw * size)),mode='nearest')

class Propogator:
    def __init__(self, config):
        video_config = vars(config)
        video_config['enable_long_term'] = not video_config['disable_long_term']
        video_config['enable_long_term_count_usage'] = (
                video_config['enable_long_term'] and
                (100
                 / (video_config['max_mid_term_frames'] - video_config['min_mid_term_frames'])
                 * video_config['num_prototypes'])
                >= video_config['max_long_term_elements']
        )
        network = XMem(video_config, "./saves/XMem.pth").cuda().eval()
        model_weights = torch.load("./saves/XMem.pth")
        network.load_weights(model_weights, init_as_zero_if_needed=True)
        processor = InferenceCore(network, config=video_config)
        im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(args.size, interpolation=InterpolationMode.BILINEAR),
        ])
        self.config=config
        self.max_step=config.max_step
        self.video_config=video_config
        self.palette = Image.open("./salt/mask_palette.png").getpalette()
        self.mapper = MaskMapper()
        self.network=network
        self.processor=processor
        self.im_transform=im_transform
        self.first_mask_loaded = False
        self.labels=None
        self.msk=None


    def clear_memory_cache(self):
        del self.mapper
        del self.processor
        torch.cuda.empty_cache()
        # video_config = vars(self.config)
        # video_config['enable_long_term'] = not video_config['disable_long_term']
        # video_config['enable_long_term_count_usage'] = (
        #         video_config['enable_long_term'] and
        #         (100
        #          / (video_config['max_mid_term_frames'] - video_config['min_mid_term_frames'])
        #          * video_config['num_prototypes'])
        #         >= video_config['max_long_term_elements']
        # )
        # self.video_config=video_config
        #
        # network = XMem(self.video_config, "./saves/XMem.pth").cuda().eval()
        # model_weights = torch.load("./saves/XMem.pth")
        # network.load_weights(model_weights, init_as_zero_if_needed=True)
        self.processor = InferenceCore(self.network, config=self.video_config)
        self.mapper=MaskMapper()

    def propagate(self,img_path,image_id):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            img = Image.open(img_path).convert('RGB')
            img = self.im_transform(img)
            rgb = img.cuda()
            # Run the model on this frame
            prob = self.processor.step(rgb, self.msk, self.labels, end=False)
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            this_out_path = os.path.join("./buffer")
            os.makedirs(this_out_path, exist_ok=True)
            out_mask = self.mapper.remap_index_mask(out_mask)
            out_img = Image.fromarray(out_mask)
            out_img = out_img.resize((1280, 720), Image.NEAREST)
            # out_img = out_img.resize((1290, 730), Image.NEAREST)
            # out_img = out_img.resize((640, 480), Image.NEAREST)
            # out_img = out_img.resize((3840, 2160), Image.NEAREST)
            # if self.palette is not None:
                # out_img.putpalette(self.palette)
            out_img.putpalette(self.palette)
            # out_img.save(os.path.join(this_out_path, str(image_id).zfill(5) + '.png'))
            out_img_data=np.array(out_img)
            out_img_data[out_img_data > 0] = 1
            # fortran_ground_truth_binary_mask = np.asfortranarray(out_img_data)
            # compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
            # compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")
            return out_img_data

    def get_mask_label(self,step):
        if step == 1:
            mask = np.array(Image.open("./salt/mask.png").convert('P'), dtype=np.uint8)
            self.msk = mask
            # print("reinitialize mask")
        else:
            self.msk = None

        if not self.first_mask_loaded:
            if self.msk is not None:
                self.first_mask_loaded = True
            else:
                # no point to do anything without a mask
                pass
        # Map possibly non-continuous labels to continuous ones
        if self.msk is not None:
            # print(self.first_mask_loaded)
            self.msk, self.labels = self.mapper.convert_mask(self.msk)
            self.msk = torch.Tensor(self.msk).cuda()
            self.msk = resize_mask(self.msk.unsqueeze(0), args.size)[0]
            self.processor.set_all_labels(list(self.mapper.remappings.values()))
            # print("reinitialize labels")
        else:
            self.labels = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="/data/labeling_tool/debug_data/video_data")
    parser.add_argument("--categories", type=str,default="massive,encrusting,branching,foliaceous,columnar,laminar,free,soft,sponge")
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements',
                        help='LT_max in paper, increase if objects disappear for a long time',
                        type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int,
                        default=-1)
    parser.add_argument('--size', default=640, type=int,
                        help='Resize the shorter side to this size. -1 to use original resolution. ')
    parser.add_argument('--max_step', default=10, type=int,help='max step to propogate')
    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
    parser.add_argument('--save_all', action='store_true',
                        help='Save all frames. Useful only in YouTubeVOS/long-time video')

    args = parser.parse_args()


    task_path = args.task_path
    dataset_path = os.path.join(task_path)
    onnx_models_path = os.path.join(task_path, "models")


    categories = None
    if args.categories is not None:
        categories = args.categories.split(",")
    
    coco_json_path = os.path.join(task_path,"annotations.json")

    editor = Editor(
        onnx_models_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )

    propogator=Propogator(config=args)

    app = QApplication(sys.argv)
    window = ApplicationInterface(app, editor,propogator)
    window.show()
    sys.exit(app.exec_())
