{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Inn1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

inn1Schema :: SQLSchema
inn1Schema =
  let columnNames = HashMap.fromList [("1", "RoomId"), ("10", "CheckIn"), ("11", "CheckOut"), ("12", "Rate"), ("13", "LastName"), ("14", "FirstName"), ("15", "Adults"), ("16", "Kids"), ("2", "roomName"), ("3", "beds"), ("4", "bedType"), ("5", "maxOccupancy"), ("6", "basePrice"), ("7", "decor"), ("8", "Code"), ("9", "Room")]
      columnTypes = HashMap.fromList [("1", ColumnType_TEXT), ("10", ColumnType_TEXT), ("11", ColumnType_TEXT), ("12", ColumnType_NUMBER), ("13", ColumnType_TEXT), ("14", ColumnType_TEXT), ("15", ColumnType_NUMBER), ("16", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("5", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_NUMBER), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "Rooms"), ("1", "Reservations")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "1"), ("13", "1"), ("14", "1"), ("15", "1"), ("16", "1"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "0"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7"]), ("1", ["8", "9", "10", "11", "12", "13", "14", "15", "16"])]
      foreignKeys = HashMap.fromList [("9", "1")]
      primaryKeys = ["1", "8"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

inn1Queries :: [Text.Text]
inn1Queries =
  [ "select count(*) from reservations as t1 join rooms as t2 on t1.room = t2.roomid",
    "select count(*) from reservations as t1 where t1.adults + t1.kids > 0",
    "select count(*) from reservations as t1 where 0 < t1.adults + t1.kids",
    "select count(*) from reservations as t1 join rooms as t2 on t1.room = t2.roomid where t2.maxoccupancy = t1.adults + t1.kids;"
  ]

inn1QueriesFails :: [Text.Text]
inn1QueriesFails = []

inn1ParserTests :: TestItem
inn1ParserTests =
  Group "inn1" $
    (ParseQueryExprWithGuardsAndTypeChecking inn1Schema <$> inn1Queries)
      <> (ParseQueryExprWithGuards inn1Schema <$> inn1Queries)
      <> (ParseQueryExprWithoutGuards inn1Schema <$> inn1Queries)
      <> (ParseQueryExprFails inn1Schema <$> inn1QueriesFails)

inn1LexerTests :: TestItem
inn1LexerTests =
  Group "inn1" $
    LexQueryExpr inn1Schema <$> inn1Queries
