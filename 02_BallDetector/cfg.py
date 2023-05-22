import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.FullLoader)
        for k in dic:
            setattr(args, k, dic[k]) ## args.k = dic[k]

def get_args():

    parser = argparse.ArgumentParser('BackHand detection used x3d model')
    parser.add_argument('--c', default='', type=str, help='config file path')
    args = parser.parse_args()

    load_yaml(args, args.c)

    return args


if __name__ == "__main__":
    args = get_args()
    print(args.target_player)