# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np

try:
    import _init_paths
    import models
except Exception as e:
    import pidnet.models as models
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# color_map = [(128, 64, 128),
#              (244, 35, 232),
#              (70, 70, 70),
#              (102, 102, 156),
#              (190, 153, 153),
#              (153, 153, 153),
#              (250, 170, 30),
#              (220, 220, 0),
#              (107, 142, 35),
#              (152, 251, 152),
#              (70, 130, 180),
#              (220, 20, 60),
#              (255, 0, 0),
#              (0, 0, 142),
#              (0, 0, 70),
#              (0, 60, 100),
#              (0, 80, 100),
#              (0, 0, 230),
#              (119, 11, 32)]

color_map = [[0, 0, 0], [255, 255, 255]]


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-l', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='dir for pretrained model',
                        default='../../checkpoints/pidnet_best.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='/home/bny220000/data/projects/data/carla/val_ood/',
                        type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)

    args = parser.parse_args()

    return args


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                       (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


if __name__ == '__main__':
    args = parse_args()

    all_dirs = []
    data_dirs = []
    rootdir = args.r

    for rootdir, dirs, files in os.walk(rootdir):
        for subdir in dirs:
            all_dirs.append(os.path.join(rootdir, subdir))

    target_dirs = ['back_camera', 'front_camera', 'right_back_camera', 'left_back_camera', 'right_front_camera',
                   'left_front_camera']

    model = models.pidnet.get_pred_model(args.a, 2)

    model = load_pretrained(model, args.p).to("cuda:1")
    for d in all_dirs:
        if d.split('/')[-1] in target_dirs:
            data_dirs.append(d)

    for sub_dir in tqdm(data_dirs):
        data_path = sub_dir + '/*' + args.t
        images_list = glob.glob(data_path)
        sv_path = "/".join(list(data_path.split('/')[0:-1])) + '_pidnet/'
        print(sv_path)
        # model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)

        model.eval()

        with torch.no_grad():
            # print(images_list)
            for img_path in images_list:
                # print(img_path)
                img_name = img_path.split("/")[-1]
                # img = cv2.imread(os.path.join(args.r, img_name),
                #                cv2.IMREAD_COLOR)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                # print(os.path.join(sub_dir, img_name))
                sv_img = np.zeros_like(img).astype(np.uint8)
                img = input_transform(img)
                img = img.transpose((2, 0, 1)).copy()
                img = torch.from_numpy(img).unsqueeze(0).to("cuda:1")
                pred = model(img)
                pred = F.interpolate(pred, size=img.size()[-2:],
                                     mode='bilinear', align_corners=True)
                pred = torch.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

                for i, color in enumerate(color_map):
                    for j in range(3):
                        sv_img[:, :, j][pred == i] = color_map[i][j]
                sv_img = Image.fromarray(sv_img)

                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                sv_img.save(sv_path + img_name)
