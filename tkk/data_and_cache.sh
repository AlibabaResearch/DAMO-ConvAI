#!/bin/sh

pip install gdown
gdown 1cwmYbypNUobZdm2LibiGAfkd7QiPrf3X
unzip tkk-files.zip
cp -a tkk-files/data data
cp -a tkk-files/cache output/cache
rm -rf tkk-files.zip
rm -rf tkk-files
rm -rf __MACOSX