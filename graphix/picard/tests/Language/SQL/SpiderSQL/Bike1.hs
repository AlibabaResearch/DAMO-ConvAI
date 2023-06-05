{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Bike1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

bike1Schema :: SQLSchema
bike1Schema =
  let columnNames = HashMap.fromList [("1", "id"), ("10", "docks_available"), ("11", "time"), ("12", "id"), ("13", "duration"), ("14", "start_date"), ("15", "start_station_name"), ("16", "start_station_id"), ("17", "end_date"), ("18", "end_station_name"), ("19", "end_station_id"), ("2", "name"), ("20", "bike_id"), ("21", "subscription_type"), ("22", "zip_code"), ("23", "date"), ("24", "max_temperature_f"), ("25", "mean_temperature_f"), ("26", "min_temperature_f"), ("27", "max_dew_point_f"), ("28", "mean_dew_point_f"), ("29", "min_dew_point_f"), ("3", "lat"), ("30", "max_humidity"), ("31", "mean_humidity"), ("32", "min_humidity"), ("33", "max_sea_level_pressure_inches"), ("34", "mean_sea_level_pressure_inches"), ("35", "min_sea_level_pressure_inches"), ("36", "max_visibility_miles"), ("37", "mean_visibility_miles"), ("38", "min_visibility_miles"), ("39", "max_wind_Speed_mph"), ("4", "long"), ("40", "mean_wind_speed_mph"), ("41", "max_gust_speed_mph"), ("42", "precipitation_inches"), ("43", "cloud_cover"), ("44", "events"), ("45", "wind_dir_degrees"), ("46", "zip_code"), ("5", "dock_count"), ("6", "city"), ("7", "installation_date"), ("8", "station_id"), ("9", "bikes_available")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_NUMBER), ("13", ColumnType_NUMBER), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_NUMBER), ("17", ColumnType_TEXT), ("18", ColumnType_TEXT), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_TEXT), ("22", ColumnType_NUMBER), ("23", ColumnType_TEXT), ("24", ColumnType_NUMBER), ("25", ColumnType_NUMBER), ("26", ColumnType_NUMBER), ("27", ColumnType_NUMBER), ("28", ColumnType_NUMBER), ("29", ColumnType_NUMBER), ("3", ColumnType_NUMBER), ("30", ColumnType_NUMBER), ("31", ColumnType_NUMBER), ("32", ColumnType_NUMBER), ("33", ColumnType_NUMBER), ("34", ColumnType_NUMBER), ("35", ColumnType_NUMBER), ("36", ColumnType_NUMBER), ("37", ColumnType_NUMBER), ("38", ColumnType_NUMBER), ("39", ColumnType_NUMBER), ("4", ColumnType_NUMBER), ("40", ColumnType_NUMBER), ("41", ColumnType_NUMBER), ("42", ColumnType_NUMBER), ("43", ColumnType_NUMBER), ("44", ColumnType_TEXT), ("45", ColumnType_NUMBER), ("46", ColumnType_NUMBER), ("5", ColumnType_NUMBER), ("6", ColumnType_TEXT), ("7", ColumnType_TEXT), ("8", ColumnType_NUMBER), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "station"), ("1", "status"), ("2", "trip"), ("3", "weather")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "2"), ("13", "2"), ("14", "2"), ("15", "2"), ("16", "2"), ("17", "2"), ("18", "2"), ("19", "2"), ("2", "0"), ("20", "2"), ("21", "2"), ("22", "2"), ("23", "3"), ("24", "3"), ("25", "3"), ("26", "3"), ("27", "3"), ("28", "3"), ("29", "3"), ("3", "0"), ("30", "3"), ("31", "3"), ("32", "3"), ("33", "3"), ("34", "3"), ("35", "3"), ("36", "3"), ("37", "3"), ("38", "3"), ("39", "3"), ("4", "0"), ("40", "3"), ("41", "3"), ("42", "3"), ("43", "3"), ("44", "3"), ("45", "3"), ("46", "3"), ("5", "0"), ("6", "0"), ("7", "0"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7"]), ("1", ["8", "9", "10", "11"]), ("2", ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]), ("3", ["23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46"])]
      foreignKeys = HashMap.fromList [("8", "1")]
      primaryKeys = ["1", "12"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

bike1Queries :: [Text.Text]
bike1Queries =
  [ "select t1.date from weather as t1 where t1.max_temperature_f > 85",
    "select weather.date from weather where weather.max_temperature_f > 85",
    "select date from weather where weather.max_temperature_f > 85",
    "select weather.date from weather where max_temperature_f > 85",
    "select date from weather where precipitation_inches > 85",
    "select date from weather where max_temperature_f > 85",
    "select date, zip_code from weather where max_temperature_f >= 80",
    "select zip_code, count(*) from weather where max_wind_speed_mph >= 25 group by zip_code",
    "select date, zip_code from weather where min_dew_point_f < (select min(min_dew_point_f) from weather where zip_code = 94107)",
    "select date, mean_temperature_f, mean_humidity from weather order by max_gust_speed_mph desc limit 3",
    "select distinct zip_code from weather except select distinct zip_code from weather where max_dew_point_f >= 70",
    "select date, max_temperature_f - min_temperature_f from weather order by max_temperature_f - min_temperature_f limit 1"
  ]

bike1QueriesFails :: [Text.Text]
bike1QueriesFails = []

bike1ParserTests :: TestItem
bike1ParserTests =
  Group "bike1" $
    (ParseQueryExprWithGuardsAndTypeChecking bike1Schema <$> bike1Queries)
      <> (ParseQueryExprWithGuards bike1Schema <$> bike1Queries)
      <> (ParseQueryExprWithoutGuards bike1Schema <$> bike1Queries)
      <> (ParseQueryExprFails bike1Schema <$> bike1QueriesFails)

bike1LexerTests :: TestItem
bike1LexerTests =
  Group "bike1" $
    LexQueryExpr bike1Schema <$> bike1Queries
