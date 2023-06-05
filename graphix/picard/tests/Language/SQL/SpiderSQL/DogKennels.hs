{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.DogKennels where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

dogKennelsSchema :: SQLSchema
dogKennelsSchema =
  let columnNames = HashMap.fromList [("1", "breed_code"), ("10", "owner_id"), ("11", "first_name"), ("12", "last_name"), ("13", "street"), ("14", "city"), ("15", "state"), ("16", "zip_code"), ("17", "email_address"), ("18", "home_phone"), ("19", "cell_number"), ("2", "breed_name"), ("20", "dog_id"), ("21", "owner_id"), ("22", "abandoned_yn"), ("23", "breed_code"), ("24", "size_code"), ("25", "name"), ("26", "age"), ("27", "date_of_birth"), ("28", "gender"), ("29", "weight"), ("3", "charge_id"), ("30", "date_arrived"), ("31", "date_adopted"), ("32", "date_departed"), ("33", "professional_id"), ("34", "role_code"), ("35", "first_name"), ("36", "street"), ("37", "city"), ("38", "state"), ("39", "zip_code"), ("4", "charge_type"), ("40", "last_name"), ("41", "email_address"), ("42", "home_phone"), ("43", "cell_number"), ("44", "treatment_id"), ("45", "dog_id"), ("46", "professional_id"), ("47", "treatment_type_code"), ("48", "date_of_treatment"), ("49", "cost_of_treatment"), ("5", "charge_amount"), ("6", "size_code"), ("7", "size_description"), ("8", "treatment_type_code"), ("9", "treatment_type_description")]
      columnTypes = HashMap.fromList [("1", ColumnType_TEXT), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_TEXT), ("17", ColumnType_TEXT), ("18", ColumnType_TEXT), ("19", ColumnType_TEXT), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_NUMBER), ("22", ColumnType_TEXT), ("23", ColumnType_TEXT), ("24", ColumnType_TEXT), ("25", ColumnType_TEXT), ("26", ColumnType_TEXT), ("27", ColumnType_TIME), ("28", ColumnType_TEXT), ("29", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("30", ColumnType_TIME), ("31", ColumnType_TIME), ("32", ColumnType_TIME), ("33", ColumnType_NUMBER), ("34", ColumnType_TEXT), ("35", ColumnType_TEXT), ("36", ColumnType_TEXT), ("37", ColumnType_TEXT), ("38", ColumnType_TEXT), ("39", ColumnType_TEXT), ("4", ColumnType_TEXT), ("40", ColumnType_TEXT), ("41", ColumnType_TEXT), ("42", ColumnType_TEXT), ("43", ColumnType_TEXT), ("44", ColumnType_NUMBER), ("45", ColumnType_NUMBER), ("46", ColumnType_NUMBER), ("47", ColumnType_TEXT), ("48", ColumnType_TIME), ("49", ColumnType_NUMBER), ("5", ColumnType_NUMBER), ("6", ColumnType_TEXT), ("7", ColumnType_TEXT), ("8", ColumnType_TEXT), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "Breeds"), ("1", "Charges"), ("2", "Sizes"), ("3", "Treatment_Types"), ("4", "Owners"), ("5", "Dogs"), ("6", "Professionals"), ("7", "Treatments")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "4"), ("11", "4"), ("12", "4"), ("13", "4"), ("14", "4"), ("15", "4"), ("16", "4"), ("17", "4"), ("18", "4"), ("19", "4"), ("2", "0"), ("20", "5"), ("21", "5"), ("22", "5"), ("23", "5"), ("24", "5"), ("25", "5"), ("26", "5"), ("27", "5"), ("28", "5"), ("29", "5"), ("3", "1"), ("30", "5"), ("31", "5"), ("32", "5"), ("33", "6"), ("34", "6"), ("35", "6"), ("36", "6"), ("37", "6"), ("38", "6"), ("39", "6"), ("4", "1"), ("40", "6"), ("41", "6"), ("42", "6"), ("43", "6"), ("44", "7"), ("45", "7"), ("46", "7"), ("47", "7"), ("48", "7"), ("49", "7"), ("5", "1"), ("6", "2"), ("7", "2"), ("8", "3"), ("9", "3")]
      tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5"]), ("2", ["6", "7"]), ("3", ["8", "9"]), ("4", ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]), ("5", ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]), ("6", ["33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"]), ("7", ["44", "45", "46", "47", "48", "49"])]
      foreignKeys = HashMap.fromList [("21", "10"), ("23", "1"), ("24", "6"), ("45", "20"), ("46", "33"), ("47", "8")]
      primaryKeys = ["1", "3", "6", "8", "10", "20", "33", "44"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

dogKennelsQueries :: [Text.Text]
dogKennelsQueries =
  [ "select first_name from professionals union select first_name from owners except select name from dogs",
    "select t1.owner_id, t1.zip_code from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id join treatments as t3 on t2.dog_id = t3.dog_id group by t1.owner_id order by sum(t3.cost_of_treatment) desc limit 1"
  ]

dogKennelsQueriesFails :: [Text.Text]
dogKennelsQueriesFails = []

dogKennelsParserTests :: TestItem
dogKennelsParserTests =
  Group "dogKennels" $
    (ParseQueryExprWithGuardsAndTypeChecking dogKennelsSchema <$> dogKennelsQueries)
      <> (ParseQueryExprWithGuards dogKennelsSchema <$> dogKennelsQueries)
      <> (ParseQueryExprWithoutGuards dogKennelsSchema <$> dogKennelsQueries)
      <> (ParseQueryExprFails dogKennelsSchema <$> dogKennelsQueriesFails)

dogKennelsLexerTests :: TestItem
dogKennelsLexerTests =
  Group "dogKennels" $
    LexQueryExpr dogKennelsSchema <$> dogKennelsQueries
