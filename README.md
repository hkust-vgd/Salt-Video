# Salt-Video
An annotation tool for marine data labeling.

## Setup
1. clone the repository
2. create a conda environment using the ```environment.yaml``` file  
    ```conda env create -f environment.yaml```
    
## preparation

1. Please download the original model weights of [SAM (ViT-H)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and the [XMeM](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth) weights and put them to ```saves``` folder.
2. Install segment anything by ```pip install git+https://github.com/facebookresearch/segment-anything.git```
3. Generate the embeddings in advance; run
```python helpers/extract_embeddings.py --dataset-path ./video_seqs/shark```
4. Generate the onnx models in advance; run
```python helpers/generate_onnx.py --dataset-path ./video_seqs/shark```
we use multi-thread to generate the onnx models. If your computer is less powerful, you can set ```--nthread``` to a smaller valuer (default is 8).

## Labeling demo
https://github.com/hkust-vgd/Salt-Video/assets/18065488/372aac02-cc46-4475-8c7e-f69475728ee0



## Usage
Activate the conda environment and run  
```python segment_anything_annotator.py --dataset-path <your_data_path>```

## Advanced features
- Multi-object tracking and segmentation (check the ```mots``` branch)
- Keyframe caption and refinement (check the ```caption``` branch)

## Acknowledgement

Our labeling tool is heavily based on 
+ [SALT](https://github.com/anuragxel/salt); The layout design and the basic logic!
+ [XMeM](https://github.com/hkchengrex/XMem); the mask propagation algorithm. 

Thanks for their contributions to the whole community.

##  Citing Salt-Video

If you find this labeling tool helpful, please consider citing:
```
@article{zhengcoralvos,
	title={CoralVOS: Dataset and Benchmark for Coral Video Segmentation},
	author={Zheng, Ziqiang and Xie, Yaofeng and Liang, Haixin and Yu, Zhibin and Yeung, Sai-Kit},
	journal={arXiv preprint arXiv:2310.01946},
	year={2023}
}
```
