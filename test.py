import os 
import cv2 
import numpy as np 
import pandas as pd 
from datasets import SteelDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse 
import time 
from pathlib import Path
from learning import Learning
from utils import load_yaml, init_seed
import importlib
import torch
import pandas as pd 
from tqdm import tqdm

sigmoid = lambda x: 1 / (1 + np.exp(-x))
class_params = {0: (0.65, 10000), 1: (0.7, 10000), 2: (0.7, 10000), 3: (0.6, 10000)}
def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS']) if config[name_package]['ARGS'] is not None else dict()
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package


def read_data(file_name):
    df = pd.read_csv(os.path.join(file_name))
    df['ImageId'], df['ClassId'] = zip(*df['Image_Label'].str.split('_'))
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    return df 
def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_config.yaml', help='train config path')
    parser.add_argument('--resume_path', type=str, default='./saved/', help='resume path')
    args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    config = load_yaml(config_folder)
    init_seed(config['SEED'])
    test_df = read_data('./data/sample_submission.csv')
    test_dataset = getattribute(config = config, name_package = 'TEST_DATASET', df = test_df)
    test_dataloader = getattribute(config = config, name_package = 'TEST_DATALOADER', dataset = test_dataset)
    model = getattribute(config = config, name_package = 'MODEL')
    print("Loading checkpoint: {} ...".format(args.resume_path))
    checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    encoded_pixels = []
    for idx, (data, _) in enumerate(tqdm(test_dataloader)):
        data = data.cuda()
        outputs = model(data)
        for probability in outputs:
            probability = probability.cpu().detach().numpy()
            print('prob: ', probability.shape)
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, (4, 350, 525), interpolation=cv2.INTER_LINEAR)
            print('prob: ', probability.shape)
            predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1
    sub = pd.read_csv('./data/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
if __name__=='__main__':
    main()

