
cd ./text_data
mv data_for_damd.json data_for_space.json
cd ../
cp -r ./text_data/* ./space_concat/space/data/multiwoz2.0/
cp -r ./text_data/* ./space_word/space/data/multiwoz2.0/
cp -r ./text_data/* ./space-3/space/data/multiwoz2.0/

cp -r ./db/* ./space_concat/db/
cp -r ./db/* ./space_word/db/
cp -r ./db/* ./space_3/db/
