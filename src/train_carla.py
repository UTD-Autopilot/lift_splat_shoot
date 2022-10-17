import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from time import time
from src.models import compile_model
from tensorboardX import SummaryWriter
from transforms3d.euler import euler2mat
from efficientnet_pytorch import EfficientNet
from src.tools import SimpleLoss, get_batch_iou, normalize_img, img_transform
from src.losses import *

import os
import json
import math
import torch


def save_pred(pred, binimg, multi, type):
    if multi:
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()[0]
        pred = pred[:, :]
        output = np.zeros(shape=(200, 200, 3))

        output[pred == 0] = (0, 0, 0)
        output[pred == 1] = (0, 0, 255)
        output[pred == 2] = (255, 0, 0)
        output[pred == 3] = (0, 255, 0)
        output[pred == 4] = (255, 0, 255)

        cv2.imwrite("pred_val_" + type + ".jpg", output)

        binimg = torch.argmax(binimg, dim=1).squeeze(0).cpu().numpy()[0]
        binimg = binimg[:, :]
        output = np.zeros(shape=(200, 200, 3))

        output[binimg == 0] = (0, 0, 0)
        output[binimg == 1] = (0, 0, 255)
        output[binimg == 2] = (255, 0, 0)
        output[binimg == 3] = (0, 255, 0)
        output[binimg == 4] = (255, 0, 255)

        cv2.imwrite("binimgs_" + type + ".jpg", output)
    else:
        cv2.imwrite("pred_" + type + ".jpg", np.array(pred.sigmoid().detach().cpu())[0, 0] * 255)
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
    def __init__(self, record_path, data_aug_conf, type="default", multi=False):
        self.record_path = record_path
        self.multi = multi
        self.type = type
        self.data_aug_conf = data_aug_conf
        self.vehicles = len(os.listdir(os.path.join(self.record_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.record_path, 'agents/0/back_camera')))

        with open(os.path.join(self.record_path, 'agents/0/sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)

    def __len__(self):
        return self.ticks * self.vehicles

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

        binimgs = Image.open(os.path.join(agent_path + "birds_view_semantic_camera", str(idx) + '.png'))

        binimgs = np.array(binimgs)

        if self.multi:
            vehicles = mask(binimgs, (0, 0, 142))
            road = mask(binimgs, (128, 64, 128))
            road_line = mask(binimgs, (157, 234, 50))
            sidewalk = mask(binimgs, (244, 35, 232))

            empty = np.ones((200, 200))
            is_empty = np.logical_or(vehicles == 1, road == 1)
            is_empty = np.logical_or(is_empty, road_line == 1)
            is_empty = np.logical_or(is_empty, sidewalk == 1)

            empty[is_empty] = 0

            binimgs = np.stack((empty, vehicles, road, road_line, sidewalk))
        else:
            binimgs = mask(binimgs, (0, 0, 142))
            binimgs = binimgs[None, :, :]

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


def get_val(model, val_loader, device, loss_fn, type):
    model.eval()

    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0

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
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'iou': total_intersect / total_union if (total_union > 0) else 1.0,
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
        pos_weight=2.13,
        logdir='./runs',
        type='default',
        multi=False,
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

    train_dataset = CarlaDataset(os.path.join(dataroot, "train/"), data_aug_conf, type=type, multi=multi)
    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, type=type, multi=multi)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                                               num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)

    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    if multi:
        model = compile_model(grid_conf, data_aug_conf, outC=5)
    else:
        model = compile_model(grid_conf, data_aug_conf, outC=1)

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

    model = nn.DataParallel(model, device_ids=gpus)
    # model.load_state_dict(torch.load("./experiments/grid_search/1/model6000.pt"), strict=False)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = SimpleLoss(pos_weight).cuda(device)
    writer = SummaryWriter(logdir=logdir)

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
    num_classes = 1

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(nepochs):
        np.random.seed()

        for batchi, (imgs, img_segs, depths, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(
                train_loader):
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
                # loss = edl_mse_loss(torch.ones(size=(16,1,200,200)).view(-1, 1), torch.ones(size=(16,1,200,200)).view(-1, 1), epoch, num_classes, 10, device)
                # print(loss)
                loss = edl_mse_loss(preds.view(-1, 1), binimgs.view(-1, 1), epoch, num_classes, 10, device)
                # loss = loss_fn(preds, binimgs)

                evidence = torch.relu(preds)
                alpha = evidence + 1
                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                # loss = torch.mean(loss)
                u = torch.mean(u)
            else:
                loss = loss_fn(preds, binimgs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)
                writer.add_scalar('train/uncertainty', u, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                print(counter, "iou:", iou)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val(model, val_loader, device, loss_fn, type)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()

            save_pred(preds, binimgs, multi, type)


