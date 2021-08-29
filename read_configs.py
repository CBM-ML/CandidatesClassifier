import numpy as np
import pandas as pd
import tomli


def read_log_vars(inp_file):
    with open(str(inp_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)

    return inp_dict['non_log_variables'], inp_dict['log_variables']


def read_peak(inp_file):
    with open(str(inp_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)
    return  inp_dict


def read_train_vars(inp_file):
    with open(str(inp_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)
    return  inp_dict['train_vars']
