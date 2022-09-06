import cv2
import torch
from time import time

from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import os
import json
import math
from transforms3d.euler import euler2mat
from PIL import Image
from .models import compile_model
from .tools import SimpleLoss, get_batch_iou, normalize_img, img_transform


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


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, record_path, data_aug_conf, ticks):
        self.record_path = record_path
        self.data_aug_conf = data_aug_conf
        self.ticks = ticks

        with open(os.path.join(self.record_path, 'sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)

    def __len__(self):
        return self.ticks

    def __getitem__(self, idx):
        imgs = []
        img_segs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        binimgs = Image.open(os.path.join(self.record_path + "birds_view_semantic_camera", str(idx) + '.png'))
        binimgs = binimgs.crop((25, 25, 175, 175))
        binimgs = binimgs.resize((200, 200))
        binimgs = np.array(binimgs)
        binimgs = torch.tensor(binimgs).permute(2, 1, 0)[0]
        binimgs = binimgs[None, :, :]/255

        for sensor_name, sensor_info in self.sensors_info['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(self.record_path + sensor_name, str(idx) + '.png'))
                image_seg = Image.open(os.path.join(self.record_path + sensor_name + "_semantic", str(idx) + '.png'))

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
                img_seg = torch.tensor(img_seg).permute(2, 0, 1)[0]
                img_seg = img_seg[None, :, :]

                imgs.append(normalize_img(img))
                # img_segs.append(normalize_img(img_seg))
                img_segs.append(img_seg/255)
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

        return (torch.stack(imgs).float(), torch.stack(img_segs).float(), torch.stack(rots).float(), torch.stack(trans).float(),
                torch.stack(intrins).float(), torch.stack(post_rots).float(), torch.stack(post_trans).float(), binimgs.float())

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
        for (imgs, img_segs, rots, trans, intrins, post_rots, post_trans, binimgs) in val_loader:

            if type == "seg":
                imgs = torch.cat((imgs, img_segs), 2)

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

            cv2.imwrite("pred_val_" + type + ".jpg", np.array(preds.sigmoid().detach().cpu())[0, 0] * 255)
            cv2.imwrite("binimgs_val_" + type + ".jpg", np.array(binimgs.detach().cpu())[0, 0] * 255)

    model.train()

    return {
        'loss': total_loss / len(val_loader.dataset),
        'iou': total_intersect / total_union,
    }


def train(
        dataroot='../data/carla',
        nepochs=10000,
        gpuid=0,

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
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 45.0, 1.0],

        bsz=8,
        val_step=2000,
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

    # 14980 1404
    train_ticks = 14980
    val_ticks = 1404

    train_dataset = CarlaDataset(os.path.join(dataroot, "train/"), data_aug_conf, train_ticks)
    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, val_ticks)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True,
                                               num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz,
                                             shuffle=False, num_workers=nworkers)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    if type == "seg":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=6)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
    writer = SummaryWriter(logdir=logdir)

    print("--------------------------------------------------")
    print(f"Starting training on {type} model")
    print(f"Batch size: {bsz} ")
    print(f"Validation interval: {val_step} ")
    print("Training using CARLA ")
    print("TRAIN LOADER: ", len(train_loader.dataset))
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, img_segs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            if type == "seg":
                imgs = torch.cat((imgs, img_segs), 2)

            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )

            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            cv2.imwrite("pred_" + type + ".jpg", np.array(preds.sigmoid().detach().cpu())[0, 0] * 255)
            cv2.imwrite("binimgs_" + type + ".jpg", np.array(binimgs.detach().cpu())[0, 0] * 255)

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

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
