{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.MuseumVisit where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

museumVisitSchema :: SQLSchema
museumVisitSchema =
  let columnNames = HashMap.fromList [("1", "Museum_ID"), ("10", "visitor_ID"), ("11", "Num_of_Ticket"), ("12", "Total_spent"), ("2", "Name"), ("3", "Num_of_Staff"), ("4", "Open_Year"), ("5", "ID"), ("6", "Name"), ("7", "Level_of_membership"), ("8", "Age"), ("9", "Museum_ID")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_NUMBER), ("12", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("4", ColumnType_NUMBER), ("5", ColumnType_NUMBER), ("6", ColumnType_TEXT), ("7", ColumnType_NUMBER), ("8", ColumnType_NUMBER), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "museum"), ("1", "visitor"), ("2", "visit")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "2"), ("11", "2"), ("12", "2"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "1"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4"]), ("1", ["5", "6", "7", "8"]), ("2", ["9", "10", "11", "12"])]
      foreignKeys = HashMap.fromList [("10", "5"), ("9", "1")]
      primaryKeys = ["1", "5", "9"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

museumVisitQueries :: [Text.Text]
museumVisitQueries =
  [ "select t1.name from visitor as t1 join visit as t2 on t1.id = t2.visitor_id join museum as t3 on t3.museum_id = t2.museum_id where t3.open_year < 2009 intersect select t1.name from visitor as t1 join visit as t2 on t1.id = t2.visitor_id join museum as t3 on t3.museum_id = t2.museum_id where t3.open_year > 2011",
    "select count(*) from museum where open_year > 2013 or open_year < 2008"
  ]

museumVisitQueriesFails :: [Text.Text]
museumVisitQueriesFails = []

museumVisitParserTests :: TestItem
museumVisitParserTests =
  Group "museumVisit" $
    (ParseQueryExprWithGuardsAndTypeChecking museumVisitSchema <$> museumVisitQueries)
      <> (ParseQueryExprWithGuards museumVisitSchema <$> museumVisitQueries)
      <> (ParseQueryExprWithoutGuards museumVisitSchema <$> museumVisitQueries)
      <> (ParseQueryExprFails museumVisitSchema <$> museumVisitQueriesFails)

museumVisitLexerTests :: TestItem
museumVisitLexerTests =
  Group "museumVisit" $
    LexQueryExpr museumVisitSchema <$> museumVisitQueries
