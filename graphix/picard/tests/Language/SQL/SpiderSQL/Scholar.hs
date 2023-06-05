{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Scholar where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

scholarSchema :: SQLSchema
scholarSchema =
  let columnNames = HashMap.fromList [("1", "venueId"), ("10", "keyphraseName"), ("11", "paperId"), ("12", "title"), ("13", "venueId"), ("14", "year"), ("15", "numCiting"), ("16", "numCitedBy"), ("17", "journalId"), ("18", "citingPaperId"), ("19", "citedPaperId"), ("2", "venueName"), ("20", "paperId"), ("21", "datasetId"), ("22", "paperId"), ("23", "keyphraseId"), ("24", "paperId"), ("25", "authorId"), ("3", "authorId"), ("4", "authorName"), ("5", "datasetId"), ("6", "datasetName"), ("7", "journalId"), ("8", "journalName"), ("9", "keyphraseId")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_NUMBER), ("12", ColumnType_TEXT), ("13", ColumnType_NUMBER), ("14", ColumnType_NUMBER), ("15", ColumnType_NUMBER), ("16", ColumnType_NUMBER), ("17", ColumnType_NUMBER), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_NUMBER), ("22", ColumnType_NUMBER), ("23", ColumnType_NUMBER), ("24", ColumnType_NUMBER), ("25", ColumnType_NUMBER), ("3", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("5", ColumnType_NUMBER), ("6", ColumnType_TEXT), ("7", ColumnType_NUMBER), ("8", ColumnType_TEXT), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "venue"), ("1", "author"), ("2", "dataset"), ("3", "journal"), ("4", "keyphrase"), ("5", "paper"), ("6", "cite"), ("7", "paperDataset"), ("8", "paperKeyphrase"), ("9", "writes")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "4"), ("11", "5"), ("12", "5"), ("13", "5"), ("14", "5"), ("15", "5"), ("16", "5"), ("17", "5"), ("18", "6"), ("19", "6"), ("2", "0"), ("20", "7"), ("21", "7"), ("22", "8"), ("23", "8"), ("24", "9"), ("25", "9"), ("3", "1"), ("4", "1"), ("5", "2"), ("6", "2"), ("7", "3"), ("8", "3"), ("9", "4")]
      tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4"]), ("2", ["5", "6"]), ("3", ["7", "8"]), ("4", ["9", "10"]), ("5", ["11", "12", "13", "14", "15", "16", "17"]), ("6", ["18", "19"]), ("7", ["20", "21"]), ("8", ["22", "23"]), ("9", ["24", "25"])]
      foreignKeys = HashMap.fromList [("13", "1"), ("17", "7"), ("18", "11"), ("19", "11"), ("22", "11"), ("23", "9"), ("24", "11"), ("25", "3")]
      primaryKeys = ["1", "3", "5", "7", "9", "11", "18", "21", "23", "24"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

scholarQueries :: [Text.Text]
scholarQueries =
  [ "select distinct t1.paperid, count ( t3.citingpaperid ) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct ( t1.paperid ) from paper as t1",
    "select distinct t1.paperid, count(t3.citingpaperid) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct (t1.paperid), count(t3.citingpaperid) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct ( t1.paperid ), count ( t3.citingpaperid ) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid join venue as t2 on t2.venueid = t1.venueid where t1.year = 2012 and t2.venuename = \"ACL\" group by t1.paperid having count ( t3.citingpaperid ) > 7;"
  ]

scholarQueriesFails :: [Text.Text]
scholarQueriesFails = []

scholarParserTests :: TestItem
scholarParserTests =
  Group "scholar" $
    (ParseQueryExprWithGuardsAndTypeChecking scholarSchema <$> scholarQueries)
      <> (ParseQueryExprWithGuards scholarSchema <$> scholarQueries)
      <> (ParseQueryExprWithoutGuards scholarSchema <$> scholarQueries)
      <> (ParseQueryExprFails scholarSchema <$> scholarQueriesFails)

scholarLexerTests :: TestItem
scholarLexerTests =
  Group "scholar" $
    LexQueryExpr scholarSchema <$> scholarQueries
