from hipe4ml.tree_handler import TreeHandler
import tomli
import sys


def convertDF(input_file):
    """
    Opens input file in toml format, retrives signal, background and deploy data
    like TreeHandler objects

    Parameters
    ------------------------------------------------
    df: str
        input toml file
    """
    with open(str(input_file), encoding="utf-8") as inp_file:
        inp_dict = tomli.load(inp_file)
    signal = TreeHandler(inp_dict["signal"]["path"], inp_dict["signal"]["tree"])
    background = TreeHandler(inp_dict["background"]["path"], inp_dict["background"]["tree"])
    deploy =  TreeHandler(inp_dict["deploy"]["path"], inp_dict["deploy"]["tree"])

    return signal, background, deploy
