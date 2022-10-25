from .train_carla import train

def search():
    weights = [1.5, 2, 2.5, 3, 4, 5]

    for weight in weights:
        train(dataroot='../data/carla/',  logdir='./experiments/grid_search_ce/' + str(weight), gpus=(4,5,6,7), weight=weight, multi=True, nepochs=10, bsz=64)