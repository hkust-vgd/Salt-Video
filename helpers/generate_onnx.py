# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of segment_anything repository and source tree.
# Adapted from onnx_model_example.ipynb in the segment_anything repository.
# Please see the original notebook for more details and other examples and additional usage.
import warnings
import os, shutil
import argparse

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

import cv2
import torch

from collections.abc import Iterable
from multiprocessing import Pool
import sys
from PIL import Image
def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)
def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True,
                            file=sys.stdout):
    """
    Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.
    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                continue
    pool.close()
    pool.join()
    return results

def save_onnx_model(checkpoint, model_type, onnx_model_path, orig_im_size, opset_version, quantize = True):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor(orig_im_size, dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if quantize:
        temp_model_path = os.path.join(os.path.split(onnx_model_path)[0], "temp.onnx")
        shutil.copy(onnx_model_path, temp_model_path)
        quantize_dynamic(
            model_input=temp_model_path,
            model_output=onnx_model_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        os.remove(temp_model_path)

def save_onnx(item):
    save_onnx_model(item['checkpoint_path'], item['model_type'], item['onnx_model_path'], item['orig_im_size'], item['opset_version'], item['quantize'])

def main(checkpoint_path, model_type, onnx_models_path, dataset_path, opset_version, quantize, nthread):
    if not os.path.exists(onnx_models_path):
        os.makedirs(onnx_models_path)
    images_path = os.path.join(dataset_path, "images")
    name_mappings={}
    for image_path in os.listdir(images_path):
        if image_path.endswith(".jpg") or image_path.endswith(".png") or image_path.endswith(".jpeg") or image_path.endswith(".JPG") or image_path.endswith(".JPEG"):
            im_path = os.path.join(images_path, image_path)
            cv2_im = cv2.imread(im_path)
            if image_path.endswith(".jpeg") or image_path.endswith(".JPEG"):
                name_img=image_path[:-5]
            else:
                name_img=image_path[:-4]
            try:
                name_mappings[name_img]=[cv2_im.shape[0],cv2_im.shape[1]]
            except:
                pass
    output=[]
    for orig_im_name in name_mappings:
        onnx_model_path = os.path.join(onnx_models_path, f"sam_onnx.{name_mappings[orig_im_name][0]}_{name_mappings[orig_im_name][1]}_{orig_im_name}.onnx")
        if os.path.exists(onnx_model_path):
            continue
        orig_im_size=[name_mappings[orig_im_name][0],name_mappings[orig_im_name][1]]
        item={}
        item['checkpoint_path']=checkpoint_path
        item['model_type']=model_type
        item['onnx_model_path']=onnx_model_path
        item['orig_im_size']=orig_im_size
        item['opset_version']=opset_version
        item['quantize']=quantize
        output.append(item)
    track_parallel_progress(save_onnx,output,nthread)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./saves/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--dataset-path", type=str, default="./video_seqs/shark")
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    dataset_path = args.dataset_path
    onnx_models_path = os.path.join(dataset_path, "models")
    opset_version = args.opset_version
    quantize = args.quantize
    nthread = args.nthread

    main(checkpoint_path, model_type, onnx_models_path, dataset_path, opset_version, quantize, nthread)
