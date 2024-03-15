# Salt-Video
An annotation tool for coral video segmentation

## Setup
1. clone the repository
2. create a conda environment using the ```environment.yml``` file  
    ```conda env create -f environment.yml```

## Usage
Activate the conda environment and run  
```python segment_anything_annotator.py --task-path <your_data_path>```


## Acknowledgement

Our labeling tool is heavily based on 
+ [SALT](https://github.com/anuragxel/salt) The layout design and the basic logic!
+ [XMeM](https://github.com/hkchengrex/XMem) the mask propagation algorithm. 

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
