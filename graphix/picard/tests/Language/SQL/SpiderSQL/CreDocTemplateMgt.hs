{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.CreDocTemplateMgt where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

creDocTemplateMgtSchema :: SQLSchema
creDocTemplateMgtSchema =
  let columnNames = HashMap.fromList [("1", "Template_Type_Code"), ("10", "Template_ID"), ("11", "Document_Name"), ("12", "Document_Description"), ("13", "Other_Details"), ("14", "Paragraph_ID"), ("15", "Document_ID"), ("16", "Paragraph_Text"), ("17", "Other_Details"), ("2", "Template_Type_Description"), ("3", "Template_ID"), ("4", "Version_Number"), ("5", "Template_Type_Code"), ("6", "Date_Effective_From"), ("7", "Date_Effective_To"), ("8", "Template_Details"), ("9", "Document_ID")]
      columnTypes = HashMap.fromList [("1", ColumnType_TEXT), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_NUMBER), ("15", ColumnType_NUMBER), ("16", ColumnType_TEXT), ("17", ColumnType_TEXT), ("2", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("4", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("6", ColumnType_TIME), ("7", ColumnType_TIME), ("8", ColumnType_TEXT), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "Ref_Template_Types"), ("1", "Templates"), ("2", "Documents"), ("3", "Paragraphs")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "2"), ("11", "2"), ("12", "2"), ("13", "2"), ("14", "3"), ("15", "3"), ("16", "3"), ("17", "3"), ("2", "0"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5", "6", "7", "8"]), ("2", ["9", "10", "11", "12", "13"]), ("3", ["14", "15", "16", "17"])]
      foreignKeys = HashMap.fromList [("10", "3"), ("15", "9"), ("5", "1")]
      primaryKeys = ["1", "3", "9", "14"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

creDocTemplateMgtQueries :: [Text.Text]
creDocTemplateMgtQueries =
  [ "select template_type_code from templates group by template_type_code having count(*) < 3"
  ]

creDocTemplateMgtQueriesFails :: [Text.Text]
creDocTemplateMgtQueriesFails = []

creDocTemplateMgtParserTests :: TestItem
creDocTemplateMgtParserTests =
  Group "creDocTemplateMgt" $
    (ParseQueryExprWithGuardsAndTypeChecking creDocTemplateMgtSchema <$> creDocTemplateMgtQueries)
      <> (ParseQueryExprWithGuards creDocTemplateMgtSchema <$> creDocTemplateMgtQueries)
      <> (ParseQueryExprWithoutGuards creDocTemplateMgtSchema <$> creDocTemplateMgtQueries)
      <> (ParseQueryExprFails creDocTemplateMgtSchema <$> creDocTemplateMgtQueriesFails)

creDocTemplateMgtLexerTests :: TestItem
creDocTemplateMgtLexerTests =
  Group "creDocTemplateMgt" $
    LexQueryExpr creDocTemplateMgtSchema <$> creDocTemplateMgtQueries
