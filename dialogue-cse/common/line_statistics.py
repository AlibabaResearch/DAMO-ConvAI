#!/usr/bin/python
# _*_coding:utf-8_*_


import os


def line_statistics(file_name):
    """
    统计文件行数
    """
    if file_name is None:
        return 0

    content = os.popen("wc -l %s" % file_name)
    line_number = int(content.read().split(" ")[0])
    return line_number
