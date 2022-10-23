import cv2
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

from src.models import compile_model
from src.train_carla import CarlaDataset, save_pred, get_iou

import os
import torch


def eval (
        dataroot='../data/carla',
        modelf='./',
        gpus=(0,),

        H=128, W=352,
        resize_lim=(0.193, 0.225),
        final_dim=(128, 352),
        bot_pct_lim=(0.0, 0.22),
        rot_lim=(-5.4, 5.4),
        rand_flip=True,

        ncams=5,
        type='default',
        multi=False,

        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        zbound=(-10.0, 10.0, 20.0),
        dbound=(4.0, 45.0, 1.0),

        bsz=16,
        nworkers=10,
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

    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, type=type, multi=multi)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)

    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    model = None
    classes = 0

    if multi:
        model = compile_model(grid_conf, data_aug_conf, outC=2)
        classes = 5
    else:
        model = compile_model(grid_conf, data_aug_conf, outC=1)
        classes = 1

    if type == "default":
        pass
    elif type == "seg" or type == "pidnet":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=4)
    elif type == "depth":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=4)
    elif type == "seg_depth":
        model.camencode.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=5)
    else:
        raise Exception("This is not a valid model type")

    model = nn.DataParallel(model, device_ids=gpus)
    model.load_state_dict(torch.load(modelf), strict=False)
    model.to(device)

    print("--------------------------------------------------")
    print(f"Starting eval on {type} model")
    print(f"Using GPUS: {gpus}")
    print("Training using CARLA ")
    print("VAL LOADER: ", len(val_loader.dataset))
    print("--------------------------------------------------")

    model.eval()

    print('running eval...')


    total_intersect = [0]*classes
    total_union = [0]*classes

    with torch.no_grad():
        for (imgs, img_segs, depths, rots, trans, intrins, post_rots, post_trans, binimgs) in tqdm(val_loader):
            print(img_segs[0].shape)

            cv2.imwrite("img_segs.png", np.array(img_segs[0, 1].permute(1, 2, 0))*255)

            if type == "seg" or type == "pidnet":
                imgs = torch.cat((imgs, img_segs), 2)
            if type == "depth":
                imgs = torch.cat((imgs, depths), 2)
            if type == "seg_depth":
                imgs = torch.cat((imgs, img_segs, depths), 2)

            preds = model(imgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # iou
            intersect, union = get_iou(preds, binimgs)

            for i in range(classes):
                total_intersect[i] += intersect[i]
                total_union[i] += union[i]

            save_pred(preds, binimgs, multi, type)

    iou = [0]*classes

    for i in range(classes):
        iou[i] = total_intersect[i]/total_union[i]

    print('iou: ' + str(iou))



