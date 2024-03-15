import os
from os import path
import json
from inference.data.video_reader import VideoReader
class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages')
                self.mask_dir = path.join(data_root, 'Annotations')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages')
            self.mask_dir = path.join(data_root, 'Annotations')
        self.size_dir = path.join(data_root, 'JPEGImages')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)