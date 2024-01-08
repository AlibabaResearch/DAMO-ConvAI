cd ../ubar
sh  data_process.sh

cd data
mkdir ../../space_baseline/text_data/
cp -r multi-woz/* ../../space_baseline/text_data/
cp -r multi-woz-analysis/* ../../space_baseline/text_data/
cp -r multi-woz-processed/* ../../space_baseline/text_data/

cd ../
cp -r db/ ../space_baseline/db/

