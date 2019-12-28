"""chictopia classes object detection."""

import os
import time
import json
import argparse
import importlib

from collections import defaultdict, OrderedDict
from multiprocessing import Pool
import tqdm

from PIL import Image as PILImage, ImageDraw
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from nnet.py_factory import NetworkFactory
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

from test.coco import apply_detection

cfg_file = 'CenterNet-52-wideeyes'
configs = json.load(open('config/' + cfg_file + '.json'))
configs["system"]["snapshot_name"] = cfg_file
system_configs.update_config(configs["system"])


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeDB:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)


def kp_decode(model, images, K, ae_threshold=0.5, kernel=3):
    with torch.no_grad():
        xs = [x.cuda(non_blocking=True) for x in images]
        detections, center = model(xs[0].unsqueeze(0), ae_threshold=ae_threshold, K=K, kernel=kernel)
    detections = detections.data.cpu().numpy()
    center = center.data.cpu().numpy()
    return detections, center


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chictopia|Yamaguchi Category Detector')
    parser.add_argument('-bpath', type=str, help='image base path')
    parser.add_argument('-text', type=str, help='text file where each row is the image filename')
    parser.add_argument('-weights', type=str, help='model weights')
    parser.add_argument('-nms_thr', help='non-maximal supression threshold (0.45)',
                        type=float, default=0.45)
    parser.add_argument('-alpha', help='threshold multiplier: alpha * opt_threshold',
                        type=float, default=0.6)
    parser.add_argument('-o', help='overwrite', action='store_true')
    parser.add_argument('-rotate', type=int, help='rotate the image with degrees', default=0)
    parser.add_argument('-bsize', type=int, help='batch size', default=1)
    parser.add_argument('-nt', type=int, help='number of threads', default=1)
    parser.add_argument('-output', type=str, help='output path',
                        default='/media/nas/tmp/rfbnet_yamaguchi')
    parser.add_argument('-light', help='use FP16', action='store_true')
    parser.add_argument('-padding', help='use padding', action='store_true')
    parser.add_argument('-benchmark', help='only do benchmark',
                        action='store_true')
    parser.add_argument('-shuffle', help='shuffle images',
                        action='store_true')
    parser.add_argument('-visualize', help='visualize detections in /tmp/',
                        action='store_true')
    parser.add_argument('-gnoise', help='gaussian noise std=1 (default=0) and mean=0',
                        default=0, type=float)
    
    args = parser.parse_args()

    module_file = "models.{}".format(system_configs.snapshot_name)
    module_file = '-'.join(module_file.split('-')[:2])
    print(module_file)
    nnet_module = importlib.import_module(module_file)

    configs['db'].update({'nms_kernel': 3})
    db = FakeDB({'configs': configs['db']})

    model = nnet_module.model(db)
    
    # torch.backends.cudnn.benchmark = True  # center net uses not-fixed image therefore CUDNN will no help
    
    snapshot = torch.load(args.weights, map_location=DEVICE)

    opt_thresholds = snapshot['optimal_thresholds']
    class_labels = snapshot['class_labels']

    weights = OrderedDict()
    for k, v in snapshot['state_dict'].items():
        weights[k.replace('module.', '')] = v
    
    model.load_state_dict(weights)
    if torch.cuda.device_count():
        model = model.cuda()

    model.eval()

    if args.light and torch.cuda.device_count() and 0:
        model = model.half()
        
    bpath = args.bpath
    files = [x.strip().replace('http://', '').replace('https://', '')
             for x in open(args.text)]

    visualize = args.visualize

    print('checking input image filenames')
    filenames = []
    with open(args.text) as fid:
        for filename in fid.readlines():
            filename = filename.strip().replace('https://', '').replace('http://', '')
            filenames.append(filename)

    # print('\nchecking output detection filenames')
    # with Pool(args.nt) as pool:
    #     # check input image files
    #     if not args.o:
    #         input_filenames = [os.path.join(args.bpath, x) for x in filenames]
    #         output = list(tqdm.tqdm(pool.imap(check_filename, input_filenames),
    #                                 total=len(input_filenames)))
    #         filenames = [u for u, o in zip(filenames, output) if o]

    # if not args.o:
    #     with Pool(args.nt) as pool:
    #         # check output detection files
    #         output_filenames = [os.path.join(args.output, x + '.json') for x in filenames]
    #         output = list(tqdm.tqdm(pool.imap(check_filename, output_filenames),
    #                                 total=len(output_filenames)))
    #         filenames2 = [u for u, o in zip(filenames, output) if not o]
            
    #         print('There are %d files, %d already exist, and %d will be computed',
    #               len(filenames), len(filenames)-len(filenames2), len(filenames2))
    #         filenames = filenames2
                                                                                    
    if args.shuffle:
        import random
        random.shuffle(filenames)

    _mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    _std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

    for filename in tqdm.tqdm(filenames):
        output_filename = os.path.join(args.output, filename + '.json')
        if os.path.exists(output_filename):
            continue
        image = cv2.imread(os.path.join(args.bpath, filename))
        if image is None:
            continue
        
        detections = apply_detection(image, model, scales=[1],
                                     decode_func=kp_decode,
                                     categories=len(class_labels),
                                     top_k=100,
                                     merge_bbox=False, nms_threshold=args.nms_thr,
                                     avg=_mean, std=_std)

        bboxes = defaultdict(list)
        for class_name, j in class_labels.items():
            if j == 0 or detections[j].size == 0:
                continue
            thr = opt_thresholds[class_name] * args.alpha
            keep_inds = (detections[j][:, -1] >= thr)
            # cat_name  = db.class_name(j)
            for bbox in detections[j][keep_inds]:
                xmin, ymin, xmax, ymax = bbox[0:4].astype(np.int32).tolist()
                score = bbox[4].item()
                # print(class_name, xmin, ymin, xmax, ymax)
                bboxes[class_name].append([xmin, ymin, xmax, ymax,
                                           [score]])
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))
        json.dump(bboxes, open(output_filename, 'w'))
