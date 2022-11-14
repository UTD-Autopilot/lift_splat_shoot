import numpy as np
import torch.nn as nn

from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

from src.models import compile_model
from src.train_carla import CarlaDataset, save_pred, get_iou, activate_uncertainty
import src.losses as l
import os
import torch

import plotly.express as px
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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
        uncertainty="entropy",

        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        zbound=(-10.0, 10.0, 20.0),
        dbound=(4.0, 45.0, 1.0),

        bsz=32,
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

    val_dataset = CarlaDataset(os.path.join(dataroot, "val/"), data_aug_conf, type=type)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    ood_val_dataset = CarlaDataset(os.path.join(dataroot, "val_ood/"), data_aug_conf, type=type)
    ood_val_loader = torch.utils.data.DataLoader(ood_val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)

    device = torch.device('cpu') if len(gpus) < 0 else torch.device(f'cuda:{gpus[0]}')

    if uncertainty is not None:
        model = compile_model(grid_conf, data_aug_conf, outC=2)
        num_classes = 2
        activation = activate_uncertainty
        uncertainty_function = getattr(l, uncertainty)
    else:
        model = compile_model(grid_conf, data_aug_conf, outC=2)
        num_classes = 2
        activation = torch.softmax
        uncertainty_function = l.entropy

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

    y_true = []
    y_scores = []

    name = type+("_"+uncertainty if uncertainty is not None else "_entropy")

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

            uncert = uncertainty_function(preds)

            try:
                preds = activation(preds)
            except Exception as e:
                preds = activation(preds, dim=1)

            intersect, union = get_iou(preds, binimgs)

            for i in range(num_classes):
                total_intersect[i] += intersect[i]
                total_union[i] += union[i]

            save_pred(preds, binimgs, type)
            plt.imsave("uncert_map"+name+".jpg", plt.cm.jet(uncert[0][0]))

            preds = preds[:, 0, :, :].ravel()
            binimgs = binimgs[:, 0, :, :].ravel()
            uncert = torch.tensor(uncert).ravel()

            vehicle = np.logical_or(preds.cpu() > 0.5, binimgs.cpu() == 1).bool()

            preds = preds[vehicle]
            binimgs = binimgs[vehicle]
            uncert = uncert[vehicle]

            pred = (preds > 0.5)
            tgt = binimgs.bool()
            intersect = (pred == tgt).type(torch.int64)

            y_true += intersect.tolist()
            # y_true += binimgs.cpu().tolist()
            uncert = -uncert
            y_scores += uncert.tolist()

    iou = [0]*num_classes

    for i in range(num_classes):
        iou[i] = total_intersect[i]/total_union[i]

    print('iou: ' + str(iou))

    print(len(y_true))
    print(len(y_scores))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.ylim([0, 1.05])
    plt.ylim([0, 1.05])

    roc_display.plot(ax=ax1, label=type)
    pr_display.plot(ax=ax2, label=type)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    plt.savefig(f"{name}_combined.png")
    print(f"pr: {pr} roc: {auc_score}")

    return roc_display, pr_display, auc_score, pr

