#!/usr/bin/env python3

import os
import json
import argparse

def parse_file_and_command(default_dict, help_dict, params_file = None, argv=None):
    parser = argparse.ArgumentParser()

    ## UNCOMMENT TO PARSE params_file (REMOVES THE HELP) before else
    # parser.add_argument('--file', type=str, default=default_parameters['file'], help=help_str['file'])
    # params_file = parser.parse_known_args()[0].file
    params_dict = {}
    if params_file is not None:
        if os.path.exists(params_file):
            with open(params_file) as f:
                params_dict = json.load(f)
        else:
            raise KeyError("input file '"+params_file+"' not found") 

    for name, default in default_dict.items():
        
        ## replace the defaults by the json file content
        try: default = params_dict[name]
        except KeyError: pass
        try: help_str = help_dict[name]
        except KeyError: help_str = 'no help available'

        if   type(default) is list:   
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

'''
TODO:
def dump(args)
def name(args, with_only=[])
'''