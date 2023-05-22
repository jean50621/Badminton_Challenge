import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_record_folder(args):

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")
    
    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "backup/", exist_ok=True)
    shutil.copy(args.c, args.save_dir+"config.yaml")

def get_args():

    parser = argparse.ArgumentParser("BallType")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()

    load_yaml(args, args.c)
    # build_record_folder(args)

    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    print(args.data_root)