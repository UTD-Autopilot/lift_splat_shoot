from .eval_carla import eval
import matplotlib.pyplot as plt
import matplotlib as mpl
def graph ():
    plt.rcParams.update({'font.size': 14})
    labels = ["LSS entropy", "LSS dissonance", "SegBEV entropy", "SegBEV dissonance"]
    uncerts = [None, "dissonance", None, "dissonance"]
    modelpaths = ["./experiments/default/model17000.pt", "./experiments/default_u/model25000.pt", "./experiments/pidnet/model37000.pt", "./experiments/pidnet_u/model26000.pt"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])

    count = 0
    for uncert in uncerts:
        modelpath = modelpaths[count]
        label = labels[count]
        roc_display, pr_display, auroc, aupr = eval(dataroot='../data/carla/', modelf=modelpath, uncertainty=uncert, type="default" if count < 2 else "pidnet", bsz=32)

        roc_display.plot(ax=ax1, label=label+ " " + str(round(auroc, 2)))
        pr_display.plot(ax=ax2, label=label+ " " + str(round(aupr, 2)))
        count += 1
        print("done")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=14, ncol=1)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, fontsize=14, ncol=1)
    plt.savefig(f"all_combined_misclassification.png", dpi=300, bbox_inches='tight')
