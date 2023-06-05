{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Geo where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

geoSchema :: SQLSchema
geoSchema =
  let columnNames = HashMap.fromList [("1", "state_name"), ("10", "state_name"), ("11", "state_name"), ("12", "border"), ("13", "state_name"), ("14", "highest_elevation"), ("15", "lowest_point"), ("16", "highest_point"), ("17", "lowest_elevation"), ("18", "lake_name"), ("19", "area"), ("2", "population"), ("20", "country_name"), ("21", "state_name"), ("22", "mountain_name"), ("23", "mountain_altitude"), ("24", "country_name"), ("25", "state_name"), ("26", "river_name"), ("27", "length"), ("28", "country_name"), ("29", "traverse"), ("3", "area"), ("4", "country_name"), ("5", "capital"), ("6", "density"), ("7", "city_name"), ("8", "population"), ("9", "country_name")]
      columnTypes = HashMap.fromList [("1", ColumnType_TEXT), ("10", ColumnType_TEXT), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_TEXT), ("17", ColumnType_NUMBER), ("18", ColumnType_TEXT), ("19", ColumnType_NUMBER), ("2", ColumnType_NUMBER), ("20", ColumnType_TEXT), ("21", ColumnType_TEXT), ("22", ColumnType_TEXT), ("23", ColumnType_NUMBER), ("24", ColumnType_TEXT), ("25", ColumnType_TEXT), ("26", ColumnType_TEXT), ("27", ColumnType_NUMBER), ("28", ColumnType_TEXT), ("29", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("5", ColumnType_TEXT), ("6", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_NUMBER), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "state"), ("1", "city"), ("2", "border_info"), ("3", "highlow"), ("4", "lake"), ("5", "mountain"), ("6", "river")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "2"), ("12", "2"), ("13", "3"), ("14", "3"), ("15", "3"), ("16", "3"), ("17", "3"), ("18", "4"), ("19", "4"), ("2", "0"), ("20", "4"), ("21", "4"), ("22", "5"), ("23", "5"), ("24", "5"), ("25", "5"), ("26", "6"), ("27", "6"), ("28", "6"), ("29", "6"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "1"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9", "10"]), ("2", ["11", "12"]), ("3", ["13", "14", "15", "16", "17"]), ("4", ["18", "19", "20", "21"]), ("5", ["22", "23", "24", "25"]), ("6", ["26", "27", "28", "29"])]
      foreignKeys = HashMap.fromList [("10", "1"), ("11", "1"), ("12", "1"), ("13", "1"), ("25", "1"), ("29", "1")]
      primaryKeys = ["1", "7", "12", "13", "22", "26"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

geoQueries :: [Text.Text]
geoQueries =
  [ "select river_name from river group by river_name order by count ( distinct traverse ) desc limit 1;",
    "select river_name from river group by ( river_name ) order by count ( distinct traverse ) desc limit 1;",
    "select t1.capital from highlow as t2 join state as t1 on t1.state_name = t2.state_name where t2.lowest_elevation = ( select min ( lowest_elevation ) from highlow );",
    "select t1.capital from highlow as t2 join state as t1 on t1.state_name = t2.state_name where t2.lowest_elevation = ( select min ( lowest_elevation ) from highlow ) ;"
  ]

geoQueriesFails :: [Text.Text]
geoQueriesFails = []

geoParserTests :: TestItem
geoParserTests =
  Group "geo" $
    (ParseQueryExprWithGuardsAndTypeChecking geoSchema <$> geoQueries)
      <> (ParseQueryExprWithGuards geoSchema <$> geoQueries)
      <> (ParseQueryExprWithoutGuards geoSchema <$> geoQueries)
      <> (ParseQueryExprFails geoSchema <$> geoQueriesFails)

geoLexerTests :: TestItem
geoLexerTests =
  Group "geo" $
    LexQueryExpr geoSchema <$> geoQueries
