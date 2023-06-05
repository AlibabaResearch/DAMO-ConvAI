{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.PokerPlayer where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

pokerPlayerSchema :: SQLSchema
pokerPlayerSchema =
  let columnNames = HashMap.fromList [("1", "Poker_Player_ID"), ("10", "Birth_Date"), ("11", "Height"), ("2", "People_ID"), ("3", "Final_Table_Made"), ("4", "Best_Finish"), ("5", "Money_Rank"), ("6", "Earnings"), ("7", "People_ID"), ("8", "Nationality"), ("9", "Name")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_NUMBER), ("2", ColumnType_NUMBER), ("3", ColumnType_NUMBER), ("4", ColumnType_NUMBER), ("5", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("7", ColumnType_NUMBER), ("8", ColumnType_TEXT), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "poker_player"), ("1", "people")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "1"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9", "10", "11"])]
      foreignKeys = HashMap.fromList [("2", "7")]
      primaryKeys = ["1", "7"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

pokerPlayerQueries :: [Text.Text]
pokerPlayerQueries =
  [ "select max(final_table_made) from poker_player where earnings < 200000"
  ]

pokerPlayerQueriesFails :: [Text.Text]
pokerPlayerQueriesFails = []

pokerPlayerParserTests :: TestItem
pokerPlayerParserTests =
  Group "pokerPlayer" $
    (ParseQueryExprWithGuardsAndTypeChecking pokerPlayerSchema <$> pokerPlayerQueries)
      <> (ParseQueryExprWithGuards pokerPlayerSchema <$> pokerPlayerQueries)
      <> (ParseQueryExprWithoutGuards pokerPlayerSchema <$> pokerPlayerQueries)
      <> (ParseQueryExprFails pokerPlayerSchema <$> pokerPlayerQueriesFails)

pokerPlayerLexerTests :: TestItem
pokerPlayerLexerTests =
  Group "pokerPlayer" $
    LexQueryExpr pokerPlayerSchema <$> pokerPlayerQueries
