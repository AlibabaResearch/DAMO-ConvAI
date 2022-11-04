#!/usr/bin/python
#_*_coding:utf-8_*_

import codecs


def dump_script(module):
    """
    将当前文件打印, 供模型分析. pyc文件无法正常打印，打印对应的python文件
    """
    file_name = module.__file__
    if file_name[-3:] == "pyc":
        file_name = file_name[0:-1]

    with codecs.open(file_name, "r", "utf-8") as f_in:
        print("\n===================dump_begin====================\n")
        print("file_path: %s" % file_name)
        for line in f_in.readlines():
            print(line.rstrip().encode("utf-8"))
        print("\n===================dump_finish====================\n")
