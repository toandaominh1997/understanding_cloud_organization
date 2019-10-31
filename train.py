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


def split_dataset(config):
    df = pd.read_csv(os.path.join(config['DATA_TRAIN']))
    df['ImageId'], df['ClassId'] = zip(*df['Image_Label'].str.split('_'))
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, valid_df

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_config.yaml', help='train config path')
    args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    train_config = load_yaml(config_folder)
    init_seed(train_config['SEED'])
    
    train_df, valid_df = split_dataset(train_config)
    train_dataset = getattribute(config = train_config, name_package = 'TRAIN_DATASET', df = train_df)
    valid_dataset = getattribute(config = train_config, name_package = 'VALID_DATASET', df = valid_df)
    train_dataloader = getattribute(config = train_config, name_package = 'TRAIN_DATALOADER', dataset = train_dataset)
    valid_dataloader = getattribute(config = train_config, name_package = 'VALID_DATALOADER', dataset = valid_dataset)
    model = getattribute(config = train_config, name_package = 'MODEL')
    criterion = getattribute(config = train_config, name_package = 'CRITERION')
    optimizer = getattribute(config = train_config, name_package= 'OPTIMIZER', params = model.parameters())
    scheduler = getattribute(config = train_config, name_package = 'SCHEDULER', optimizer = optimizer)
    device = train_config['DEVICE']
    num_epoch = train_config['NUM_EPOCH']
    gradient_clipping = train_config['GRADIENT_CLIPPING']
    gradient_accumulation_steps = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config['VALIDATION_FREQUENCY']
    saved_period = train_config['SAVED_PERIOD']
    checkpoint_dir = Path(train_config['CHECKPOINT_DIR'], type(model).__name__)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    resume_path = train_config['RESUME_PATH']
    learning = Learning(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        num_epoch=num_epoch,
                        scheduler = scheduler,
                        grad_clipping = gradient_clipping,
                        grad_accumulation = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        save_period = saved_period,
                        checkpoint_dir = checkpoint_dir,
                        resume_path=resume_path)
    learning.train(tqdm(train_dataloader), tqdm(valid_dataloader))

if __name__ == "__main__":
    main()