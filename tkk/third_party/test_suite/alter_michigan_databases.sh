#~/usr/bin/env bash
set -e

DATABASE_DIR=.

copy_databases () {
  db=$1
  # Copy to *_test directory
  altered=$DATABASE_DIR/${db}_test
  cp -r "$DATABASE_DIR/$db" "$altered"

  # Rename .sqlite files
  cd "$altered"
  for f in ${db}*.sqlite
  do
    mv "$f" "${db}_test${f#${db}}"
  done
  cd -
}

alter_yelp () {
  for f in `ls $DATABASE_DIR/yelp_test/*.sqlite`
  do
    echo "ALTER TABLE neighbourhood RENAME TO neighborhood" | sqlite3 "$f"
    echo "ALTER TABLE neighborhood RENAME COLUMN neighbourhood_name TO neighborhood_name" | sqlite3 "$f"
  done
}

alter_imdb () {
  for f in `ls $DATABASE_DIR/imdb_test/*.sqlite`
  do
    echo "ALTER TABLE cast RENAME TO cast2" | sqlite3 "$f"
  done
}

alter_academic () {
  :
}

alter_geo () {
  :
}

alter_scholar () {
  :
}

# geo is an exception in that we want to change the name from "geography" to "geo_test"  
# it is easiest to achieve this is by copying "geography" to "geo" first
if [ ! -d $DATABASE_DIR/geo ] 
then 
  cp -r $DATABASE_DIR/geography $DATABASE_DIR/geo
  mv $DATABASE_DIR/geo/geography.sqlite $DATABASE_DIR/geo/geo.sqlite
fi  

for DB in imdb yelp academic geo scholar
do
  echo $DB
  if [ ! -d "$DATABASE_DIR/${DB}_test" ]
  then
    copy_databases $DB
    alter_"$DB"
  else
    echo "$DATABASE_DIR/${DB}_test already exists"
  fi
done
