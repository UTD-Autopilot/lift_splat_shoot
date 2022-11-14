
import os
import cv2
import numpy as np
import sklearn
import torch
from PIL import Image
from sklearn.utils import class_weight

from .base_dataset import BaseDataset


class CarlaDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=4,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=352, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(CarlaDataset, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = os.path.join(root)
        self.list_path = list_path # train or val
        self.num_classes = num_classes

        self.cameras = [
            "front_camera",
            "left_front_camera",
            "right_front_camera",
            "back_camera",
            "left_back_camera",
            "right_back_camera",
        ]

        self.vehicles = len(os.listdir(os.path.join(self.root, self.list_path, 'agents')))

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.files = self.read_files()

        self.ignore_label = ignore_label
        
        # self.color_list = [[0, 0, 0], [0, 0, 142], [128, 64, 128], (81,  0, 81)]
        self.color_list = [[0, 0, 0], [0, 0, 142], [157, 234, 50]]

        self.class_weights = None

        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        for vehicle in range(0, self.vehicles):
            for camera in self.cameras:
                folder = os.path.join(self.root, self.list_path, "agents", str(vehicle), camera)
                semantic_folder = os.path.join(self.root, self.list_path, "agents", str(vehicle), camera+"_semantic")

                for filename in os.listdir(folder):
                    if not filename.endswith(".png"):
                        continue
                    name = os.path.splitext(os.path.basename(filename))[0]
                    files.append({
                        "img": os.path.join(folder, filename),
                        "label": os.path.join(semantic_folder, filename),
                        "name": name
                    })

        print(len(files))
        return files
        
    def color2label(self, color_map):
        #label = np.ones(color_map.shape[:2])*self.ignore_label
        label = np.zeros(color_map.shape[:2], dtype=np.uint8)
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(item["img"]).convert('RGB')
        image = np.array(image)
        size = image.shape

        color_map = Image.open(item["label"]).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
