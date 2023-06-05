{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Singer where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

singerSchema :: SQLSchema
singerSchema =
  let columnNames = HashMap.fromList [("1", "Singer_ID"), ("10", "Highest_Position"), ("2", "Name"), ("3", "Birth_Year"), ("4", "Net_Worth_Millions"), ("5", "Citizenship"), ("6", "Song_ID"), ("7", "Title"), ("8", "Singer_ID"), ("9", "Sales")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("4", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("6", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_NUMBER), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "singer"), ("1", "song")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5"]), ("1", ["6", "7", "8", "9", "10"])]
      foreignKeys = HashMap.fromList [("8", "1")]
      primaryKeys = ["1", "6"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

singerQueries :: [Text.Text]
singerQueries =
  [ "select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955",
    "select singer.citizenship from singer where singer.birth_year < 1945 intersect select singer.citizenship from singer where singer.birth_year > 1955"
  ]

singerQueriesFails :: [Text.Text]
singerQueriesFails = []

singerParserTests :: TestItem
singerParserTests =
  Group "singer" $
    (ParseQueryExprWithGuardsAndTypeChecking singerSchema <$> singerQueries)
      <> (ParseQueryExprWithGuards singerSchema <$> singerQueries)
      <> (ParseQueryExprWithoutGuards singerSchema <$> singerQueries)
      <> (ParseQueryExprFails singerSchema <$> singerQueriesFails)

singerLexerTests :: TestItem
singerLexerTests =
  Group "singer" $
    LexQueryExpr singerSchema <$> singerQueries
