if [ ! -e out_clean.zip ];
then
    echo "downloading out_clean ..."
    wget -c http://url/to/dataset/out_clean.zip
else
    echo "out_clean has been downloaded."
fi

if [ ! -d out_clean ];
then
    echo "unziping out_clean"
    unzip out_clean.zip
else
    echo "out_clean has been unzipped"
fi

# preprocess
python explore.py --data_dir ./out_clean --output_dir ./output --max_instance_num -1

# fewshot sampling
python stat_category.py --source_dir ./output --output_dir ./output
python partition.py --source_dir ./output --output_dir ./output
python match.py --source_dir ./output --output_dir ./output --step 100
python rearrange_dataset.py --source_dir ./output --output_dir ./output

# generate dataset
python noise.py --output_dir ./output --all_file rearrange_all.json
python change_data_format_for_relation.py -d ./output
ln -s ../../../ours/output ../converted_data/text2spotasoc/relation/ours
