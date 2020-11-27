#!/usr/bin/env python3

import os
import sys
import json
import argparse

try:
    import wandb
    import pandas as pd 
    USE_WANDB = True
except ModuleNotFoundError:
    USE_WANDB = False


def parse_file_and_command(default_dict, help_dict):
    parser = argparse.ArgumentParser()

    params_dict = {}
    
    if len(sys.argv)>1:
        args_file = sys.argv[1] 
        if args_file[-5:]=='.json':
            if os.path.exists(args_file):
                with open(args_file) as f:
                    params_dict = json.load(f)
                print(f'using parameters from file {args_file}')
            else:
                raise ValueError(f'file {args_file} not found')
        else:
            print('using default parameters with args')
    else:
        print('using default parameters')

    for name, default in default_dict.items():
        
        ## replace the defaults by the json file content
        try: default = params_dict[name]
        except KeyError: pass
        try: help_str = help_dict[name]
        except KeyError: help_str = 'no help available'

        if type(default) is list:
            if type(default[0]) is list:
                parser.add_argument('--'+name, type=json.loads, default=default, help=help_str)
            else:
                parser.add_argument('--'+name, nargs='+', type=type(default[0]), default=default, help=help_str)
        elif type(default) is bool:
            parser.add_argument('--'+name, action='store_true', default=default, help=help_str)
            parser.add_argument('--no_'+name, dest=name, action='store_false', default=default, help='disables '+name)  
        else:
            parser.add_argument('--'+name, type=type(default), default=default, help=help_str)
    
    # using parser.parse_known_args()[0] instead of parser.parse_args() preserve
    # compatibility with jupyter in vscode
    return parser.parse_known_args()[0]


class ObjectView():
    def __init__(self,dict):
        self.__dict__.update(dict)


if USE_WANDB: 

    def wandb_download_table(project_name):
        api = wandb.Api()
        runs = api.runs(project_name)
        summary_df = pd.DataFrame.from_records([
            {k:v for k,v in run.summary.items() if not k.startswith('gradients/')}
            for run in runs
        ]) 
        config_df = pd.DataFrame.from_records([
            {k:v for k,v in run.config.items() if not k.startswith('_')}
            for run in runs 
        ])  
        wandb_df = pd.DataFrame({
            'wandb_name': [run.name for run in runs],
            'wandb_id': [run.id for run in runs],
            'wandb_state': [run.state for run in runs]
        }) 
        return pd.concat([wandb_df, config_df, summary_df], axis=1)
