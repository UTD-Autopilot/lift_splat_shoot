import cv2
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

import matplotlib.pyplot as plt
from src.models import compile_model
from src.train_carla import CarlaDataset, save_pred, get_iou, activate_uncertainty
from src.losses import entropy_softmax, dissonance, vacuity
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
        uncertainty=False,

        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        zbound=(-10.0, 10.0, 20.0),
        dbound=(4.0, 45.0, 1.0),

        bsz=1,
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers)

    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    if uncertainty:
        model = compile_model(grid_conf, data_aug_conf, outC=2)
        num_classes = 2
        activation = activate_uncertainty
    elif multi:
        model = compile_model(grid_conf, data_aug_conf, outC=2)
        num_classes = 2
        activation = torch.softmax
    else:
        model = compile_model(grid_conf, data_aug_conf, outC=1)
        num_classes = 1
        activation = torch.sigmoid

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

    total_intersect = [0]*num_classes
    total_union = [0]*num_classes

    uncert = []
    iou = []

    with torch.no_grad():
        for (imgs, img_segs, depths, rots, trans, intrins, post_rots, post_trans, binimgs) in tqdm(val_loader):
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

            try:
                preds = activation(preds)
            except Exception as e:
                preds = activation(preds, dim=1)

            # iou
            intersect, union = get_iou(preds, binimgs)

            for i in range(num_classes):
                total_intersect[i] += intersect[i]
                total_union[i] += union[i]

            if uncertainty:
                try:
                    map = dissonance(np.array(preds.cpu()))
                    map = map / np.max(map)
                    cv2.imwrite("uncert_map_u.jpg", map[0][0] * 255)
                    uncert.append(np.mean(dissonance(np.array(preds.cpu()))))
                    iou.append(intersect[0]/union[0])
                except Exception as e:
                    iou.append(0)
            else:
                try:
                    map = entropy_softmax(np.array(preds.cpu()))[0]
                    map = map / np.max(map)
                    cv2.imwrite("uncert_map.jpg", map[0][0] * 255)
                    # print(map.shape)
                    uncert.append(np.mean(entropy_softmax(np.array(preds.cpu()))[0]))
                    iou.append(intersect[0]/union[0])
                except Exception as e:
                    iou.append(0)

            save_pred(preds, binimgs, type+str(multi)+str(uncertainty))

    # iou = [0]*num_classes
    #
    # for i in range(num_classes):
    #     iou[i] = total_intersect[i]/total_union[i]

    # print('iou: ' + str(iou))

    plt.scatter(uncert, iou)
    plt.xlabel("Uncertainty")
    plt.ylabel("IOU")
    plt.savefig("u_plot.jpg")


