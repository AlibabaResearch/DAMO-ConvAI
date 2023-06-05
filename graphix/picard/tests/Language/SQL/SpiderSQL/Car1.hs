{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Car1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

car1Schema :: SQLSchema
car1Schema =
  let columnNames = HashMap.fromList [("1", "ContId"), ("10", "ModelId"), ("11", "Maker"), ("12", "Model"), ("13", "MakeId"), ("14", "Model"), ("15", "Make"), ("16", "Id"), ("17", "MPG"), ("18", "Cylinders"), ("19", "Edispl"), ("2", "Continent"), ("20", "Horsepower"), ("21", "Weight"), ("22", "Accelerate"), ("23", "Year"), ("3", "CountryId"), ("4", "CountryName"), ("5", "Continent"), ("6", "Id"), ("7", "Maker"), ("8", "FullName"), ("9", "Country")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_NUMBER), ("12", ColumnType_TEXT), ("13", ColumnType_NUMBER), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_NUMBER), ("17", ColumnType_NUMBER), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_NUMBER), ("22", ColumnType_NUMBER), ("23", ColumnType_NUMBER), ("3", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("5", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_TEXT), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "continents"), ("1", "countries"), ("2", "car_makers"), ("3", "model_list"), ("4", "car_names"), ("5", "cars_data")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "3"), ("11", "3"), ("12", "3"), ("13", "4"), ("14", "4"), ("15", "4"), ("16", "5"), ("17", "5"), ("18", "5"), ("19", "5"), ("2", "0"), ("20", "5"), ("21", "5"), ("22", "5"), ("23", "5"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "2"), ("7", "2"), ("8", "2"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5"]), ("2", ["6", "7", "8", "9"]), ("3", ["10", "11", "12"]), ("4", ["13", "14", "15"]), ("5", ["16", "17", "18", "19", "20", "21", "22", "23"])]
      foreignKeys = HashMap.fromList [("11", "6"), ("14", "12"), ("16", "13"), ("5", "1"), ("9", "3")]
      primaryKeys = ["1", "3", "6", "10", "13", "16"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

car1Queries :: [Text.Text]
car1Queries =
  [ "select *",
    "select 1",
    "select * from countries",
    "select countryid from countries",
    "select t1.countryid, t1.countryname from countries as t1",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t2.id",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t3.maker = t3.maker",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t3.maker",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t3.maker where t3.model = \"Fiat\"",
    "select maker from model_list",
    "select t3.maker from model_list as t3",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t3.maker where t3.model = \"Fiat\"",
    "select modelid from model_list",
    "select modelid from model_list where modelid in (select modelid from model_list)",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))))))",
    "SELECT count(*) FROM CONTINENTS;",
    "SELECT count(*) FROM CONTINENTS;",
    "SELECT T1.ContId ,  T1.Continent ,  count(*) FROM CONTINENTS AS T1 JOIN COUNTRIES AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId;",
    "SELECT T1.ContId ,  T1.Continent ,  count(*) FROM CONTINENTS AS T1 JOIN COUNTRIES AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId;",
    "SELECT count(*) FROM COUNTRIES;",
    "SELECT count(*) FROM COUNTRIES;",
    "SELECT T1.FullName ,  T1.Id ,  count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id;",
    "SELECT T1.FullName ,  T1.Id ,  count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id;",
    "SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.horsepower ASC LIMIT 1;",
    "SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.horsepower ASC LIMIT 1;",
    "SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Weight  <  (SELECT avg(Weight) FROM CARS_DATA)",
    "SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Weight  <  (SELECT avg(Weight) FROM CARS_DATA)",
    "SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  1970;",
    "SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  1970;",
    "SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA);",
    "SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA);",
    "SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.model  =  T2.model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.id WHERE T3.year  >  1980;",
    "SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.model  =  T2.model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.id WHERE T3.year  >  1980;",
    "SELECT T1.Continent ,  count(*) FROM CONTINENTS AS T1 JOIN COUNTRIES AS T2 ON T1.ContId  =  T2.continent JOIN car_makers AS T3 ON T2.CountryId  =  T3.Country GROUP BY T1.Continent;",
    "SELECT T1.Continent ,  count(*) FROM CONTINENTS AS T1 JOIN COUNTRIES AS T2 ON T1.ContId  =  T2.continent JOIN car_makers AS T3 ON T2.CountryId  =  T3.Country GROUP BY T1.Continent;",
    "SELECT T2.CountryName FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId GROUP BY T1.Country ORDER BY Count(*) DESC LIMIT 1;",
    "SELECT T2.CountryName FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId GROUP BY T1.Country ORDER BY Count(*) DESC LIMIT 1;",
    "SELECT count(*) ,  t2.fullname from model_list as t1 join car_makers as t2 on t1.maker  =  t2.id group by t2.id;",
    "SELECT Count(*) ,  T2.FullName ,  T2.id FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id GROUP BY T2.id;",
    "SELECT T1.Accelerate FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Make  =  'amc hornet sportabout (sw)';",
    "SELECT T1.Accelerate FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Make  =  'amc hornet sportabout (sw)';",
    "SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france';",
    "SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france';",
    "SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa';",
    "SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa';",
    "SELECT avg(mpg) FROM CARS_DATA WHERE Cylinders  =  4;",
    "SELECT avg(mpg) FROM CARS_DATA WHERE Cylinders  =  4;",
    "SELECT min(weight) from cars_data where cylinders  =  8 and year  =  1974",
    "SELECT min(weight) from cars_data where cylinders  =  8 and year  =  1974",
    "SELECT Maker ,  Model FROM MODEL_LIST;",
    "SELECT Maker ,  Model FROM MODEL_LIST;",
    "SELECT T1.CountryName ,  T1.CountryId FROM COUNTRIES AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.CountryId HAVING count(*)  >=  1;",
    "SELECT T1.CountryName ,  T1.CountryId FROM COUNTRIES AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.CountryId HAVING count(*)  >=  1;",
    "SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150;",
    "SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150;",
    "SELECT avg(Weight) ,  YEAR FROM CARS_DATA GROUP BY YEAR;",
    "SELECT avg(Weight) ,  YEAR FROM CARS_DATA GROUP BY YEAR;",
    "SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3;",
    "SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3;",
    "SELECT T2.horsepower ,  T1.Make FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.cylinders  =  3 ORDER BY T2.horsepower DESC LIMIT 1;",
    "SELECT T2.horsepower ,  T1.Make FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.cylinders  =  3 ORDER BY T2.horsepower DESC LIMIT 1;",
    "SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.mpg DESC LIMIT 1;",
    "SELECT t1.model from car_names as t1 join cars_data as t2 on t1.makeid  =  t2.id order by t2.mpg desc limit 1;",
    "SELECT avg(horsepower) FROM CARS_DATA WHERE YEAR  <  1980;",
    "SELECT avg(horsepower) from cars_data where year  <  1980;",
    "SELECT avg(T2.edispl) FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T1.Model  =  'volvo';",
    "SELECT avg(T2.edispl) FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T1.Model  =  'volvo';",
    "SELECT max(Accelerate) ,  Cylinders FROM CARS_DATA GROUP BY Cylinders;",
    "SELECT max(Accelerate) ,  Cylinders FROM CARS_DATA GROUP BY Cylinders;",
    "SELECT Model FROM CAR_NAMES GROUP BY Model ORDER BY count(*) DESC LIMIT 1;",
    "SELECT Model FROM CAR_NAMES GROUP BY Model ORDER BY count(*) DESC LIMIT 1;",
    "SELECT count(*) FROM CARS_DATA WHERE Cylinders  >  4;",
    "SELECT count(*) FROM CARS_DATA WHERE Cylinders  >  4;",
    "SELECT count(*) FROM CARS_DATA WHERE YEAR  =  1980;",
    "SELECT count(*) FROM CARS_DATA WHERE YEAR  =  1980;",
    "SELECT count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker WHERE T1.FullName  =  'American Motor Company';",
    "SELECT count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker WHERE T1.FullName  =  'American Motor Company';",
    "SELECT T1.FullName ,  T1.Id FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >  3;",
    "SELECT T1.FullName ,  T1.Id FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >  3;",
    "SELECT DISTINCT T2.Model FROM CAR_NAMES AS T1 JOIN MODEL_LIST AS T2 ON T1.Model  =  T2.Model JOIN CAR_MAKERS AS T3 ON T2.Maker  =  T3.Id JOIN CARS_DATA AS T4 ON T1.MakeId  =  T4.Id WHERE T3.FullName  =  'General Motors' OR T4.weight  >  3500;",
    "SELECT DISTINCT T2.Model FROM CAR_NAMES AS T1 JOIN MODEL_LIST AS T2 ON T1.Model  =  T2.Model JOIN CAR_MAKERS AS T3 ON T2.Maker  =  T3.Id JOIN CARS_DATA AS T4 ON T1.MakeId  =  T4.Id WHERE T3.FullName  =  'General Motors' OR T4.weight  >  3500;",
    "SELECT distinct year from cars_data where weight between 3000 and 4000;",
    "SELECT distinct year from cars_data where weight between 3000 and 4000;",
    "SELECT T1.horsepower FROM CARS_DATA AS T1 ORDER BY T1.accelerate DESC LIMIT 1;",
    "SELECT T1.horsepower FROM CARS_DATA AS T1 ORDER BY T1.accelerate DESC LIMIT 1;",
    "SELECT T1.cylinders FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Model  =  'volvo' ORDER BY T1.accelerate ASC LIMIT 1;",
    "SELECT T1.cylinders FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Model  =  'volvo' ORDER BY T1.accelerate ASC LIMIT 1;",
    "SELECT COUNT(*) FROM CARS_DATA WHERE Accelerate  >  ( SELECT Accelerate FROM CARS_DATA ORDER BY Horsepower DESC LIMIT 1 );",
    "SELECT COUNT(*) FROM CARS_DATA WHERE Accelerate  >  ( SELECT Accelerate FROM CARS_DATA ORDER BY Horsepower DESC LIMIT 1 );",
    "SELECT count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2",
    "SELECT count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2",
    "SELECT COUNT(*) FROM CARS_DATA WHERE Cylinders  >  6;",
    "SELECT COUNT(*) FROM CARS_DATA WHERE Cylinders  >  6;",
    "SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Cylinders  =  4 ORDER BY T2.horsepower DESC LIMIT 1;",
    "SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Cylinders  =  4 ORDER BY T2.horsepower DESC LIMIT 1;",
    "SELECT T2.MakeId ,  T2.Make FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Horsepower  >  (SELECT min(Horsepower) FROM CARS_DATA) AND T1.Cylinders  <=  3;",
    "SELECT t2.makeid ,  t2.make from cars_data as t1 join car_names as t2 on t1.id  =  t2.makeid where t1.horsepower  >  (select min(horsepower) from cars_data) and t1.cylinders  <  4;",
    "SELECT max(mpg) from cars_data where cylinders  =  8 or year  <  1980",
    "SELECT max(mpg) from cars_data where cylinders  =  8 or year  <  1980",
    "SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.Model  =  T2.Model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.Id JOIN CAR_MAKERS AS T4 ON T1.Maker  =  T4.Id WHERE T3.weight  <  3500 AND T4.FullName != 'Ford Motor Company';",
    "SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.Model  =  T2.Model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.Id JOIN CAR_MAKERS AS T4 ON T1.Maker  =  T4.Id WHERE T3.weight  <  3500 AND T4.FullName != 'Ford Motor Company';",
    "SELECT CountryName FROM countries EXCEPT SELECT T1.CountryName FROM countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.countryId  =  T2.Country;",
    "SELECT CountryName FROM countries EXCEPT SELECT T1.CountryName FROM countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.countryId  =  T2.Country;",
    "SELECT t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3;",
    "SELECT T1.Id ,  T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >=  2 INTERSECT SELECT T1.Id ,  T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model GROUP BY T1.Id HAVING count(*)  >  3;",
    "SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.countryId HAVING count(*)  >  3 UNION SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country JOIN MODEL_LIST AS T3 ON T2.Id  =  T3.Maker WHERE T3.Model  =  'fiat';",
    "SELECT t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  3 union select t1.countryid ,  t1.countryname from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country join model_list as t3 on t2.id  =  t3.maker where t3.model  =  'fiat';"
  ]

car1QueriesFails :: [Text.Text]
car1QueriesFails =
  [ "select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country",
    "select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country where t2.make = \"Fiat\"",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country where t2.make = \"Fiat\"",
    "select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data where horsepower = (select max(horsepower) from cars_data where year = (select max(horsepower) from cars_data where accelerate = (select max(accelerate) from cars_data where horses",
    "select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data where horsepower = (select max(horsepower) from cars_data where model = (select id from cars_data where model = (select id from cars_data where model = (select id from cars_data where model = (select id from cars_data where horsepower = (select max(h"
  ]

car1QueriesFailsTypeChecking :: [Text.Text]
car1QueriesFailsTypeChecking =
  [ "select t1.countryid, t1.countryname from countries union select t1.countryname, t1.countryid from countries"
  ]

car1ParserTests :: TestItem
car1ParserTests =
  Group "car1" $
    (ParseQueryExprWithGuardsAndTypeChecking car1Schema <$> car1Queries)
      <> (ParseQueryExprWithGuards car1Schema <$> car1Queries)
      <> (ParseQueryExprWithoutGuards car1Schema <$> car1Queries)
      <> (ParseQueryExprFails car1Schema <$> car1QueriesFails)
      <> (ParseQueryExprFailsTypeChecking car1Schema <$> car1QueriesFailsTypeChecking)

car1LexerTests :: TestItem
car1LexerTests =
  Group "car1" $
    LexQueryExpr car1Schema <$> car1Queries
