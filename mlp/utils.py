#!/usr/bin/env python3

import os
import json
import argparse

def parse_file_and_command(default_dict, help_dict, params_file = 'params.json', argv=None):
    parser = argparse.ArgumentParser()

    ## WARNING: ENABLING PARSING OF THE PARAMS FILENAME REMOVES THE HELP
    # parser.add_argument('--file', type=str, default=default_parameters['file'], help=help_str['file'])
    # params_file = parser.parse_known_args()[0].file
    params_dict = {}
    if (params_file is not None):
        if os.path.exists(params_file):
            with open(params_file) as f:
                params_dict = json.load(f)
        else:
            print("warning: input file '"+params_file+"' not found") 
        

    for name, default in default_dict.items():
        
        ## replace the defaults by the json file content
        try: default = params_dict[name]
        except KeyError: pass

        ## build the kwargs that will be passed to parser.add_argument()
        kwargs = {}
        kwargs['default'] = default
        try:
            kwargs['help'] = help_dict[name]
        except KeyError:
            kwargs['help'] = 'sorry, no help string available'
        if   type(default) is list:
            kwargs['nargs']  = '+' 
            kwargs['type']   = type(default[0])
        elif type(default) is bool:
            kwargs['action'] = 'store_true'
        else:
            kwargs['type']   = type(default)
        
        ## create the script argument
        parser.add_argument('--'+name, **kwargs)

        ## create the anti-argument (for bool arguments)
        if type(default) is bool:
            parser.add_argument(
                '--no_'+name, dest=name, action='store_false', 
                default=default, help='disables '+name
            )  
    
    # using parser.parse_known_args()[0] instead of parser.parse_args() 
    # preserve compatibility with jupyter in vscode
    return parser.parse_known_args()[0]

class ObjectView():
    def __init__(self,dict):
        self.__dict__.update(dict)

'''
TODO:
def dump(args)
def name(args, with_only=[])
'''