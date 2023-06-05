{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.ConcertSinger where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

concertSingerSchema :: SQLSchema
concertSingerSchema =
  let columnNames = HashMap.fromList [("1", "Stadium_ID"), ("10", "Country"), ("11", "Song_Name"), ("12", "Song_release_year"), ("13", "Age"), ("14", "Is_male"), ("15", "concert_ID"), ("16", "concert_Name"), ("17", "Theme"), ("18", "Stadium_ID"), ("19", "Year"), ("2", "Location"), ("20", "concert_ID"), ("21", "Singer_ID"), ("3", "Name"), ("4", "Capacity"), ("5", "Highest"), ("6", "Lowest"), ("7", "Average"), ("8", "Singer_ID"), ("9", "Name")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_NUMBER), ("14", ColumnType_OTHERS), ("15", ColumnType_NUMBER), ("16", ColumnType_TEXT), ("17", ColumnType_TEXT), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_NUMBER), ("3", ColumnType_TEXT), ("4", ColumnType_NUMBER), ("5", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("7", ColumnType_NUMBER), ("8", ColumnType_NUMBER), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "stadium"), ("1", "singer"), ("2", "concert"), ("3", "singer_in_concert")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "1"), ("13", "1"), ("14", "1"), ("15", "2"), ("16", "2"), ("17", "2"), ("18", "2"), ("19", "2"), ("2", "0"), ("20", "3"), ("21", "3"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "0"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7"]), ("1", ["8", "9", "10", "11", "12", "13", "14"]), ("2", ["15", "16", "17", "18", "19"]), ("3", ["20", "21"])]
      foreignKeys = HashMap.fromList [("18", "1"), ("20", "15"), ("21", "8")]
      primaryKeys = ["1", "8", "15", "20"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

concertSingerQueries :: [Text.Text]
concertSingerQueries =
  [ "SELECT *",
    "SELECT * FROM singer",
    "SELECT count(*)",
    "SELECT count(*) FROM singer",
    "SELECT name FROM singer",
    "SELECT singer.name FROM singer",
    "SELECT T1.name FROM singer as T1",
    "SELECT age FROM singer",
    "SELECT singer.age FROM singer",
    "SELECT T1.age FROM singer as T1",
    "SELECT Song_Name FROM singer",
    "SELECT singer.Song_Name FROM singer",
    "SELECT T1.Song_Name FROM singer as T1",
    "SELECT country FROM singer",
    "SELECT singer.country FROM singer",
    "SELECT T1.country FROM singer as T1",
    "SELECT T1.name, T1.country, T1.age FROM singer AS T1 ORDER BY T1.age DESC",
    "SELECT age FROM singer ORDER BY age DESC",
    "SELECT name, age FROM singer ORDER BY age DESC",
    "SELECT name ,  country ,  age FROM singer ORDER BY age DESC",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
    "SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1",
    "SELECT DISTINCT country FROM singer WHERE age  >  20",
    "SELECT country ,  count(*) FROM singer GROUP BY country",
    "SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)",
    "SELECT LOCATION ,  name FROM stadium WHERE capacity BETWEEN 5000 AND 10000",
    "select max(capacity), average from stadium",
    "select avg(capacity) ,  max(capacity) from stadium",
    "SELECT name ,  capacity FROM stadium ORDER BY average DESC LIMIT 1",
    "SELECT count(*) FROM concert WHERE YEAR  =  2014 OR YEAR  =  2015",
    "SELECT stadium_id FROM concert",
    "SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id",
    "SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id",
    "SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1",
    "select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1",
    "SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
    "SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30",
    "SELECT name FROM stadium EXCEPT SELECT T2.name FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  =  2014",
    "SELECT T2.concert_name ,  T2.theme ,  count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id  =  T2.concert_id GROUP BY T2.concert_id",
    "select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id",
    "SELECT T2.name ,  count(*) FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id GROUP BY T2.singer_id",
    "SELECT T2.name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.singer_id  =  T2.singer_id JOIN concert AS T3 ON T1.concert_id  =  T3.concert_id WHERE T3.year  =  2014",
    "SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'",
    "SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2014 INTERSECT SELECT T2.name ,  T2.location FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.Year  =  2015",
    "select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)",
    "SELECT T1.name ,  count(*) FROM stadium AS T1 JOIN concert AS T2 ON T1.stadium_id  =  T2.stadium_id",
    "SELECT T2.name ,  count(*) FROM stadium AS T2 JOIN concert AS T1 ON T1.stadium_id  =  T2.stadium_id",
    "SELECT name FROM singer WHERE singer_id IN (SELECT singer_id FROM singer)",
    "SELECT T1.name FROM singer AS T1 JOIN singer as T2",
    "SELECT T2.name FROM singer AS T1 JOIN singer as T2",
    "SELECT name FROM singer AS T1",
    "SELECT name FROM (SELECT name FROM singer)",
    "SELECT DISTINCT name FROM (SELECT DISTINCT name FROM singer)",
    "SELECT DISTINCT name FROM (SELECT DISTINCT name FROM singer) AS T1",
    "SELECT DISTINCT T1.name FROM (SELECT DISTINCT name FROM singer) AS T1",
    "SELECT DISTINCT T1.name FROM (SELECT DISTINCT T1.name FROM singer AS T1) AS T1",
    "SELECT DISTINCT T1.name FROM (SELECT DISTINCT T2.name FROM singer AS T2) AS T1"
  ]

concertSingerQueriesFails :: [Text.Text]
concertSingerQueriesFails =
  [ "SELECT name",
    "SELECT stadium_id",
    "SELECT name FROM concert",
    "SELECT name FROM invalid",
    "SELECT * FROM concert JOIN concert",
    "SELECT * FROM concert as T1 JOIN concert as T1",
    "SELECT name FROM singer AS T1 JOIN singer as T2",
    "SELECT name FROM singer JOIN stadium",
    "select _1.* from concert as _1",
    "SELECT DISTINCT T2.Name FROM (SELECT DISTINCT T2.Name FROM singer as T2) as T1",
    "SELECT DISTINCT T1.Name FROM (SELECT DISTINCT T1.Name FROM singer as T2) as T1",
    "SELECT DISTINCT T1.Name FROM (SELECT DISTINCT T2.Name FROM singer as T2 WHERE T1.Name = T2.Name) as T1",
    "SELECT DISTINCT T1.Name FROM (SELECT DISTINCT T2.Name FROM singer as T2) as T1 WHERE T1.Name = T2.Name"
  ]

concertSingerParserTests :: TestItem
concertSingerParserTests =
  Group "concertSinger" $
    (ParseQueryExprWithGuardsAndTypeChecking concertSingerSchema <$> concertSingerQueries)
      <> (ParseQueryExprWithGuards concertSingerSchema <$> concertSingerQueries)
      <> (ParseQueryExprWithoutGuards concertSingerSchema <$> concertSingerQueries)
      <> (ParseQueryExprFails concertSingerSchema <$> concertSingerQueriesFails)

concertSingerLexerTests :: TestItem
concertSingerLexerTests =
  Group "concertSinger" $
    LexQueryExpr concertSingerSchema <$> concertSingerQueries
