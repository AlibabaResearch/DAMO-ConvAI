{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Wta1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

wta1Schema :: SQLSchema
wta1Schema =
  let columnNames = HashMap.fromList [("1", "player_id"), ("10", "loser_entry"), ("11", "loser_hand"), ("12", "loser_ht"), ("13", "loser_id"), ("14", "loser_ioc"), ("15", "loser_name"), ("16", "loser_rank"), ("17", "loser_rank_points"), ("18", "loser_seed"), ("19", "match_num"), ("2", "first_name"), ("20", "minutes"), ("21", "round"), ("22", "score"), ("23", "surface"), ("24", "tourney_date"), ("25", "tourney_id"), ("26", "tourney_level"), ("27", "tourney_name"), ("28", "winner_age"), ("29", "winner_entry"), ("3", "last_name"), ("30", "winner_hand"), ("31", "winner_ht"), ("32", "winner_id"), ("33", "winner_ioc"), ("34", "winner_name"), ("35", "winner_rank"), ("36", "winner_rank_points"), ("37", "winner_seed"), ("38", "year"), ("39", "ranking_date"), ("4", "hand"), ("40", "ranking"), ("41", "player_id"), ("42", "ranking_points"), ("43", "tours"), ("5", "birth_date"), ("6", "country_code"), ("7", "best_of"), ("8", "draw_size"), ("9", "loser_age")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_TEXT), ("12", ColumnType_NUMBER), ("13", ColumnType_NUMBER), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_NUMBER), ("17", ColumnType_NUMBER), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_TEXT), ("22", ColumnType_TEXT), ("23", ColumnType_TEXT), ("24", ColumnType_TIME), ("25", ColumnType_TEXT), ("26", ColumnType_TEXT), ("27", ColumnType_TEXT), ("28", ColumnType_NUMBER), ("29", ColumnType_TEXT), ("3", ColumnType_TEXT), ("30", ColumnType_TEXT), ("31", ColumnType_NUMBER), ("32", ColumnType_NUMBER), ("33", ColumnType_TEXT), ("34", ColumnType_TEXT), ("35", ColumnType_NUMBER), ("36", ColumnType_NUMBER), ("37", ColumnType_NUMBER), ("38", ColumnType_NUMBER), ("39", ColumnType_TIME), ("4", ColumnType_TEXT), ("40", ColumnType_NUMBER), ("41", ColumnType_NUMBER), ("42", ColumnType_NUMBER), ("43", ColumnType_NUMBER), ("5", ColumnType_TIME), ("6", ColumnType_TEXT), ("7", ColumnType_NUMBER), ("8", ColumnType_NUMBER), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "players"), ("1", "matches"), ("2", "rankings")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "1"), ("13", "1"), ("14", "1"), ("15", "1"), ("16", "1"), ("17", "1"), ("18", "1"), ("19", "1"), ("2", "0"), ("20", "1"), ("21", "1"), ("22", "1"), ("23", "1"), ("24", "1"), ("25", "1"), ("26", "1"), ("27", "1"), ("28", "1"), ("29", "1"), ("3", "0"), ("30", "1"), ("31", "1"), ("32", "1"), ("33", "1"), ("34", "1"), ("35", "1"), ("36", "1"), ("37", "1"), ("38", "1"), ("39", "2"), ("4", "0"), ("40", "2"), ("41", "2"), ("42", "2"), ("43", "2"), ("5", "0"), ("6", "0"), ("7", "1"), ("8", "1"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38"]), ("2", ["39", "40", "41", "42", "43"])]
      foreignKeys = HashMap.fromList [("13", "1"), ("32", "1"), ("41", "1")]
      primaryKeys = ["1"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

wta1Queries :: [Text.Text]
wta1Queries =
  [ "select matches.winner_name, matches.winner_rank_points from matches group by matches.winner_name order by count ( * ) desc limit 1"
  ]

wta1QueriesFails :: [Text.Text]
wta1QueriesFails = []

wta1ParserTests :: TestItem
wta1ParserTests =
  Group "wta1" $
    (ParseQueryExprWithGuardsAndTypeChecking wta1Schema <$> wta1Queries)
      <> (ParseQueryExprWithGuards wta1Schema <$> wta1Queries)
      <> (ParseQueryExprWithoutGuards wta1Schema <$> wta1Queries)
      <> (ParseQueryExprFails wta1Schema <$> wta1QueriesFails)

wta1LexerTests :: TestItem
wta1LexerTests =
  Group "wta1" $
    LexQueryExpr wta1Schema <$> wta1Queries
