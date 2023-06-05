{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Flight1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

flight1Schema :: SQLSchema
flight1Schema =
  let columnNames = HashMap.fromList [("1", "flno"), ("10", "name"), ("11", "distance"), ("12", "eid"), ("13", "name"), ("14", "salary"), ("15", "eid"), ("16", "aid"), ("2", "origin"), ("3", "destination"), ("4", "distance"), ("5", "departure_date"), ("6", "arrival_date"), ("7", "price"), ("8", "aid"), ("9", "aid")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_NUMBER), ("12", ColumnType_NUMBER), ("13", ColumnType_TEXT), ("14", ColumnType_NUMBER), ("15", ColumnType_NUMBER), ("16", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("3", ColumnType_TEXT), ("4", ColumnType_NUMBER), ("5", ColumnType_TIME), ("6", ColumnType_TIME), ("7", ColumnType_NUMBER), ("8", ColumnType_NUMBER), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "flight"), ("1", "aircraft"), ("2", "employee"), ("3", "certificate")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "2"), ("13", "2"), ("14", "2"), ("15", "3"), ("16", "3"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "0"), ("8", "0"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7", "8"]), ("1", ["9", "10", "11"]), ("2", ["12", "13", "14"]), ("3", ["15", "16"])]
      foreignKeys = HashMap.fromList [("15", "12"), ("16", "9"), ("8", "9")]
      primaryKeys = ["1", "9", "12", "15"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

flight1Queries :: [Text.Text]
flight1Queries =
  [ "select * from aircraft having count(*) >= 5",
    "select t2.name from certificate as t1 join aircraft as t2 on t2.aid = t1.aid where t2.distance > 5000 group by t1.aid having count(*) >= 5"
    -- "select t2.name from certificate as t1 join aircraft as t2 on t2.aid = t1.aid where t2.distance > 5000 group by t1.aid order by count(*) >= 5"
  ]

flight1QueriesFails :: [Text.Text]
flight1QueriesFails = []

flight1ParserTests :: TestItem
flight1ParserTests =
  Group "flight1" $
    (ParseQueryExprWithGuardsAndTypeChecking flight1Schema <$> flight1Queries)
      <> (ParseQueryExprWithGuards flight1Schema <$> flight1Queries)
      <> (ParseQueryExprWithoutGuards flight1Schema <$> flight1Queries)
      <> (ParseQueryExprFails flight1Schema <$> flight1QueriesFails)

flight1LexerTests :: TestItem
flight1LexerTests =
  Group "flight1" $
    LexQueryExpr flight1Schema <$> flight1Queries
