# coding=utf-8
# Copyright 2017 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

import xmunmt.models.rnnsearch
import xmunmt.models.vnmt


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return xmunmt.models.rnnsearch.RNNsearch
    if name == "vnmt":
        return xmunmt.models.vnmt.VNMT
    else:
        raise LookupError("Unknown model %s" % name)
