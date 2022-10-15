from .train_carla import train

def search():
    weights = [0.2, 0.5, 0.8]

    for weight in weights:
        train(dataroot='../data/carla/',  logdir='./experiments/grid_search/' + str(weight), gpus=(4,5,6,7), pos_weight=weight, nepochs=10, bsz=64)