import os 
# os.system('pip install -r requirements.txt')
import argparse 
import time 
from pathlib import Path
from learning import Learning
from utils import load_yaml, init_seed
import importlib
import torch
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models import accuracy_dice_score

def split_dataset(file_name):
    df = pd.read_csv(os.path.join(file_name))
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df['im_id'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    return df, train_ids, valid_ids

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS']) if config[name_package]['ARGS'] is not None else dict()
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_config.yaml', help='train config path')
    args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    config = load_yaml(config_folder)
    init_seed(config['SEED'])
    
    df, train_ids, valid_ids = split_dataset(config['DATA_TRAIN'])
    train_dataset = getattribute(config = config, name_package = 'TRAIN_DATASET', df = df, img_ids = train_ids)
    valid_dataset = getattribute(config = config, name_package = 'VALID_DATASET', df = df, img_ids = valid_ids)
    train_dataloader = getattribute(config = config, name_package = 'TRAIN_DATALOADER', dataset = train_dataset)
    valid_dataloader = getattribute(config = config, name_package = 'VALID_DATALOADER', dataset = valid_dataset)
    model = getattribute(config = config, name_package = 'MODEL')
    criterion = getattribute(config = config, name_package = 'CRITERION')
    optimizer = getattribute(config = config, name_package= 'OPTIMIZER', params = model.parameters())
    scheduler = getattribute(config = config, name_package = 'SCHEDULER', optimizer = optimizer)
    device = config['DEVICE']
    metric_ftns = [accuracy_dice_score]
    num_epoch = config['NUM_EPOCH']
    gradient_clipping = config['GRADIENT_CLIPPING']
    gradient_accumulation_steps = config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = config['EARLY_STOPPING']
    validation_frequency = config['VALIDATION_FREQUENCY']
    saved_period = config['SAVED_PERIOD']
    checkpoint_dir = Path(config['CHECKPOINT_DIR'], type(model).__name__)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    resume_path = config['RESUME_PATH']
    learning = Learning(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        metric_ftns=metric_ftns,
                        num_epoch=num_epoch,
                        scheduler = scheduler,
                        grad_clipping = gradient_clipping,
                        grad_accumulation_steps = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        save_period = saved_period,
                        checkpoint_dir = checkpoint_dir,
                        resume_path=resume_path)
    learning.train(tqdm(train_dataloader), tqdm(valid_dataloader))

if __name__ == "__main__":
    main() 
