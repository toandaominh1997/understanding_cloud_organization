import yaml
import numpy as np
import torch
import os
import requests
import socket
import datetime
import json

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config

def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class Slacksender(object):
    def __init__(self, webhook_url = 'https://hooks.slack.com/services/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', channel = 'pytoan'):
        self.webhook_url = webhook_url
        self.channel = channel 
        self.dump = {
            "username": "Knock Knock",
            "channel": self.channel,
            "icon_emoji": ":clapper:",
        }
        self.host_name = socket.gethostname()
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @staticmethod
    def normalize(message):
        if message is None:
            return []
        res = ['Message:\n']
        if type(message)==str:
            res.append(str('    ' + message+'\n'))
        elif type(message)==dict:
            for key, value in message.items():
                res.append('    {:15s}: {}\n'.format(str(key), value))
        
        return res

    def start(self, message = None):
        self.start_time = datetime.datetime.now()
        contents = ['Your training has started 🎬\n',
                    'Machine name: {}\n'.format(self.host_name),
                    'Starting date: {}\n'.format(self.start_time.strftime(self.DATE_FORMAT))]
        contents.extend(self.normalize(message))
        self.dump['text'] = ''.join(contents)
        self.dump['icon_emoji'] = ':clapper:'
        requests.post(self.webhook_url, json.dumps(self.dump))
    
    def process(self, message = None):
        contents = self.normalize(message)
        self.dump['text'] = ''.join(contents)
        self.dump['icon_emoji'] = ':clapper:'
        requests.post(self.webhook_url, json.dumps(self.dump))
    def end(self, message = None):
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        contents = ["Your training is complete 🎉\n",
                    'Machine name: {}\n'.format(self.host_name),
                    'Starting date: {}\n'.format(self.start_time.strftime(self.DATE_FORMAT)),
                    'End date: {}\n'.format(end_time.strftime(self.DATE_FORMAT)),
                    'Training duration: {}\n'.format(str(elapsed_time))]
        contents.extend(self.normalize(message))
        self.dump['text'] = ''.join(contents)
        self.dump['icon_emoji'] = ':tada:'
        requests.post(self.webhook_url, json.dumps(self.dump))

# mess = {}
# print(type(mess))
slack_sender = Slacksender()
slack_sender.start()
slack_sender.process({"kaka": 10})
slack_sender.end({'training kaka lala ': 1000})
