import yaml
import os
import shutil
import argparse
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_scheduler(optimizer, warm_epochs: int, max_epochs: int, train_batchs: int, lr_min: float, warmup_lr_init: float):
    n_iter_per_epoch = train_batchs
    num_steps = int(max_epochs * n_iter_per_epoch)
    warmup_steps = int(warm_epochs * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps),
            lr_min=lr_min,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=True,
        )

    return lr_scheduler


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

def get_args(with_deepspeed: bool=False):

    parser = argparse.ArgumentParser("efficient video object detection (VOD) training")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()

    load_yaml(args, args.c)
    build_record_folder(args)

    return args
