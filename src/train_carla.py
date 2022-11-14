import cv2
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from PIL import Image
from time import time

from src.models import compile_model
from tensorboardX import SummaryWriter
from transforms3d.euler import euler2mat
from efficientnet_pytorch import EfficientNet
from src.tools import normalize_img, img_transform
from src.losses import *

import os
import json
import math
import torch


def get_iou(preds, binimgs):
    classes = preds.shape[1]

    intersect = [0]*classes
    union = [0]*classes

    with torch.no_grad():
        for i in range(classes):
            pred = (preds[:, i, :, :] > 0.5)
            tgt = binimgs[:, i, :, :].bool()
            intersect[i] = (pred & tgt).sum().float().item()
            union[i] = (pred | tgt).sum().float().item()

    return intersect, union


def save_pred(pred, binimg, type):
    cv2.imwrite("pred_" + type + ".jpg", np.array(pred.detach().cpu())[0, 0] * 255)
    cv2.imwrite("binimgs_" + type + ".jpg", np.array(binimg.detach().cpu())[0, 0] * 255)


def get_camera_info(translation, rotation, sensor_options):
    roll = math.radians(rotation[2] - 90)
    pitch = -math.radians(rotation[1])
    yaw = -math.radians(rotation[0])
    rotation_matrix = euler2mat(roll, pitch, yaw)

    calibration = np.identity(3)
    calibration[0, 2] = sensor_options['image_size_x'] / 2.0
    calibration[1, 2] = sensor_options['image_size_y'] / 2.0
    calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
            2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))

    return torch.tensor(rotation_matrix), torch.tensor(translation), torch.tensor(calibration)


def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, record_path, data_aug_conf, type="default"):
        self.record_path = record_path
        self.type = type
        self.data_aug_conf = data_aug_conf
        self.vehicles = len(os.listdir(os.path.join(self.record_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.record_path, 'agents/0/back_camera')))

        with open(os.path.join(self.record_path, 'agents/0/sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)

    def __len__(self):
        return self.vehicles*self.ticks

    def __getitem__(self, idx):
        agent_number = math.floor(idx / self.ticks)
        agent_path = os.path.join(self.record_path, f"agents/{agent_number}/")

        idx = idx % self.ticks

        imgs = []
        img_segs = []
        depths = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        binimgs = np.array(Image.open(os.path.join(agent_path + "birds_view_semantic_camera", str(idx) + '.png')))

        vehicles = mask(binimgs, (0, 0, 142))
        empty = np.ones((200, 200))

        empty[vehicles == 1] = 0

        binimgs = np.stack((vehicles, empty))


        binimgs = torch.tensor(binimgs)

        for sensor_name, sensor_info in self.sensors_info['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, str(idx) + '.png'))

                if self.type == "pidnet":
                    image_seg = Image.open(os.path.join(agent_path + sensor_name + "_pidnet", str(idx) + '.png'))
                else:
                    image_seg = Image.open(os.path.join(agent_path + sensor_name + "_semantic", str(idx) + '.png'))

                depth = Image.open(os.path.join(agent_path + sensor_name + "_depth", str(idx) + '.png'))

                tran = sensor_info["transform"]["location"]
                rot = sensor_info["transform"]["rotation"]
                sensor_options = sensor_info["sensor_options"]

                rot, tran, intrin = get_camera_info(tran, rot, sensor_options)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                img_seg, _, _ = img_transform(image_seg, post_rot, post_tran,
                                              resize=resize,
                                              resize_dims=resize_dims,
                                              crop=crop,
                                              flip=flip,
                                              rotate=rotate, )

                depth, _, _ = img_transform(depth, post_rot, post_tran,
                                            resize=resize,
                                            resize_dims=resize_dims,
                                            crop=crop,
                                            flip=flip,
                                            rotate=rotate, )

                img, post_rot2, post_tran2 = img_transform(image, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate, )

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                img_seg = np.array(img_seg)

                if self.type == "pidnet":
                    img_seg = mask(img_seg, (255, 255, 255))
                else:
                    img_seg = mask(img_seg, (0, 0, 142))

                img_seg = torch.tensor(img_seg)[None, :, :]

                depth = np.array(depth)
                depth = depth[:, :, 0] + depth[:, :, 1] * 256 + depth[:, :, 2] * 256 * 256
                depth = depth / (256 * 256 * 256 - 1)
                depth = depth * 1000

                if np.max(depth) > 0:
                    depth = depth / np.max(depth)

                depth = torch.tensor(depth)[None, :, :]

                imgs.append(normalize_img(img))
                img_segs.append(img_seg)
                depths.append(depth)

                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

        return (torch.stack(imgs).float(), torch.stack(img_segs).float(), torch.stack(depths).float(),
                torch.stack(rots).float(), torch.stack(trans).float(),
                torch.stack(intrins).float(), torch.stack(post_rots).float(), torch.stack(post_trans).float(),
                binimgs.float())

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0

        return resize, resize_dims, crop, flip, rotate


def get_val(model, val_loader, device, loss_fn, type, uncertainty, activation, num_classes):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0

    print('running eval...')

    with torch.no_grad():
        for (imgs, img_segs, depths, rots, trans, intrins, post_rots, post_trans, binimgs) in tqdm(val_loader):

            if type == "seg" or type == "pidnet":
                imgs = torch.cat((imgs, img_segs), 2)
            if type == "depth":
                imgs = torch.cat((imgs, depths), 2)
            if type == "seg_depth" or type == "pidnet_depth":
                imgs = torch.cat((imgs, img_segs, depths), 2)

            preds = model(imgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            if uncertainty:
                loss = loss_fn(preds.view(-1, num_classes), binimgs.view(-1, num_classes), 0, num_classes, 10, device)
                total_loss += loss
            else:
                total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            try:
                preds = activation(preds)
            except Exception as e:
                preds = activation(preds, dim=1)

            # iou
            intersection, union = get_iou(preds, binimgs)
            iou = (intersection[0] / union[0]) * preds.shape[0]
            total_iou += iou

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'iou': total_iou / len(val_loader.dataset),
    }


def train(
        dataroot='../data/carla',
        nepochs=10000,
        gpus=(0,),

        H=128, W=352,
        resize_lim=(0.193, 0.225),
        final_dim=(128, 352),
        bot_pct_lim=(0.0, 0.22),
        rot_lim=(-5.4, 5.4),
        rand_flip=True,

        ncams=5,
        max_grad_norm=5.0,
        weight=1,
        logdir='./runs',
        type='default',
        uncertainty=False,

        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        zbound=(-10.0, 10.0, 20.0),
        dbound=(4.0, 45.0, 1.0),

        bsz=16,
        val_step=1000,
        nworkers=10,
        lr=1e-3,
        weight_decay=1e-7,
):

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }

    train_dataset = CarlaDataset(os.path.join(dataroot, "train/"), data_aug_conf, type=type)
    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, type=type)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                                               num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)

    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')
    num_classes = 2

    if uncertainty:
        activation = activate_uncertainty
    else:
        activation = torch.softmax

    model = compile_model(grid_conf, data_aug_conf, outC=num_classes)

    if type == "default":
        pass
    elif type == "seg" or type == "pidnet":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=4)
    elif type == "depth":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=4)
    elif type == "seg_depth" or type == "pidnet_depth":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=5)
    else:
        raise Exception("This is not a valid model type")

    model = nn.DataParallel(model, device_ids=gpus).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if uncertainty:
        loss_fn = edl_digamma_loss
    else:
        loss_fn = CrossEntropyLoss(weight=torch.tensor([weight, 1.0]).cuda(device))

    writer = SummaryWriter(logdir=logdir)
    print(device)
    print("--------------------------------------------------")
    print(f"Starting training on {type} model")
    print(f"Using GPUS: {gpus}")
    print(f"Batch size: {bsz} ")
    print(f"Validation interval: {val_step} ")
    print("Training using CARLA ")
    print("TRAIN LOADER: ", len(train_loader.dataset))
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    model.train()
    counter = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(nepochs):
        for batchi, (imgs, img_segs, depths, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            if type == "seg" or type == "pidnet":
                imgs = torch.cat((imgs, img_segs), 2)
            if type == "depth":
                imgs = torch.cat((imgs, depths), 2)
            if type == "seg_depth" or type == "pidnet_depth":
                imgs = torch.cat((imgs, img_segs, depths), 2)

            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )

            binimgs = binimgs.to(device)

            if uncertainty:
                loss = loss_fn(preds.view(-1, num_classes), binimgs.view(-1, num_classes), epoch, num_classes, 10, device)
            else:
                loss = loss_fn(preds, binimgs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            try:
                preds = activation(preds)
            except Exception as e:
                preds = activation(preds, dim=1)

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                intersection, union = get_iou(preds, binimgs)

                iou = (intersection[0] / union[0])

                print(counter, "iou:", iou)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val(model, val_loader, device, loss_fn, type, uncertainty, activation, num_classes)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

            save_pred(preds, binimgs, type)
