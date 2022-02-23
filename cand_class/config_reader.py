from hipe4ml.tree_handler import TreeHandler
import tomli
import sys


def convertDF(input_file, mass_var):
    """
    Opens input file in toml format, retrives signal, background and deploy data
    like TreeHandler objects

    Parameters
    ------------------------------------------------
    df: str
        input toml file
    """
    with open(str(input_file), encoding="utf-8") as inp_file:
        inp_info = tomli.load(inp_file)

    signal = TreeHandler(inp_info["signal"]["path"], inp_info["signal"]["tree"])
    background = TreeHandler(inp_info["background"]["path"], inp_info["background"]["tree"])





    bgr_left_edge = inp_info["peak_range"]["bgr_left_edge"]
    bgr_right_edge = inp_info["peak_range"]["bgr_right_edge"]

    peak_left_edge = inp_info["peak_range"]["sgn_left_edge"]
    peak_right_edge = inp_info["peak_range"]["sgn_right_edge"]


    selection = str(bgr_left_edge)+'< '+mass_var+' <'+str(peak_left_edge)+' or '+str(peak_right_edge)+\
    '< '+mass_var+' <'+str(bgr_right_edge)

    signalH = signal.get_subset(size = inp_info["number_of_events"]["number_of_signal_events"])
    bkgH = background.get_subset(selection, size=inp_info["number_of_events"]["number_of_background_events"])

    return signalH, bkgH


def read_log_vars(inp_file):
    with open(str(inp_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)

    return inp_dict['non_log_scale']['variables'], inp_dict['log_scale']['variables']



def read_train_vars(inp_file):
    with open(str(inp_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)
    return  inp_dict['train_vars']
