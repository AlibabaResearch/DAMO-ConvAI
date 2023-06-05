{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Orchestra where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

orchestraSchema :: SQLSchema
orchestraSchema =
  let columnNames = HashMap.fromList [("1", "Conductor_ID"), ("10", "Year_of_Founded"), ("11", "Major_Record_Format"), ("12", "Performance_ID"), ("13", "Orchestra_ID"), ("14", "Type"), ("15", "Date"), ("16", "Official_ratings_(millions)"), ("17", "Weekly_rank"), ("18", "Share"), ("19", "Show_ID"), ("2", "Name"), ("20", "Performance_ID"), ("21", "If_first_show"), ("22", "Result"), ("23", "Attendance"), ("3", "Age"), ("4", "Nationality"), ("5", "Year_of_Work"), ("6", "Orchestra_ID"), ("7", "Orchestra"), ("8", "Conductor_ID"), ("9", "Record_Company")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_NUMBER), ("13", ColumnType_NUMBER), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_NUMBER), ("17", ColumnType_TEXT), ("18", ColumnType_TEXT), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_OTHERS), ("22", ColumnType_TEXT), ("23", ColumnType_NUMBER), ("3", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("5", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_NUMBER), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "conductor"), ("1", "orchestra"), ("2", "performance"), ("3", "show")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "2"), ("13", "2"), ("14", "2"), ("15", "2"), ("16", "2"), ("17", "2"), ("18", "2"), ("19", "3"), ("2", "0"), ("20", "3"), ("21", "3"), ("22", "3"), ("23", "3"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5"]), ("1", ["6", "7", "8", "9", "10", "11"]), ("2", ["12", "13", "14", "15", "16", "17", "18"]), ("3", ["19", "20", "21", "22", "23"])]
      foreignKeys = HashMap.fromList [("13", "6"), ("20", "12"), ("8", "1")]
      primaryKeys = ["1", "6", "12"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

orchestraQueries :: [Text.Text]
orchestraQueries =
  [ "select orchestra.record_company from orchestra order by orchestra.year_of_founded desc"
  ]

orchestraQueriesFails :: [Text.Text]
orchestraQueriesFails = []

orchestraParserTests :: TestItem
orchestraParserTests =
  Group "orchestra" $
    (ParseQueryExprWithGuardsAndTypeChecking orchestraSchema <$> orchestraQueries)
      <> (ParseQueryExprWithGuards orchestraSchema <$> orchestraQueries)
      <> (ParseQueryExprWithoutGuards orchestraSchema <$> orchestraQueries)
      <> (ParseQueryExprFails orchestraSchema <$> orchestraQueriesFails)

orchestraLexerTests :: TestItem
orchestraLexerTests =
  Group "orchestra" $
    LexQueryExpr orchestraSchema <$> orchestraQueries
