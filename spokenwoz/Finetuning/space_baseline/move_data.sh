
cd ./text_data
mv data_for_damd.json data_for_space.json
cd ../
mkdir ./space_concat/space/data/multiwoz2.0/
mkdir ./space_word/space/data/multiwoz2.0/
mkdir ./space-3/space/data/multiwoz2.0/
cp -r ./text_data/* ./space_concat/space/data/multiwoz2.0/
cp -r ./text_data/* ./space_word/space/data/multiwoz2.0/
cp -r ./text_data/* ./space-3/space/data/multiwoz2.0/

mkdir ./space_concat/db/
mkdir ./space_word/db/
mkdir ./space-3/db/

cp -r ./db/* ./space_concat/db/
cp -r ./db/* ./space_word/db/
cp -r ./db/* ./space-3/db/
