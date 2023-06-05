{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.AssetsMaintenance where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

assetsMaintenanceSchema :: SQLSchema
assetsMaintenanceSchema =
  let columnNames = HashMap.fromList [("51", "fault_status"), ("15", "other_part_details"), ("37", "last_name"), ("48", "contact_staff_id"), ("61", "part_fault_id"), ("7", "maintenance_contract_company_id"), ("25", "supplier_company_id"), ("43", "recorded_by_staff_id"), ("28", "asset_model"), ("57", "fault_short_name"), ("13", "chargeable_yn"), ("31", "other_asset_details"), ("14", "chargeable_amount"), ("36", "first_name"), ("49", "engineer_id"), ("50", "fault_log_entry_id"), ("22", "other_staff_details"), ("19", "staff_id"), ("44", "fault_log_entry_datetime"), ("29", "asset_acquired_date"), ("56", "part_id"), ("12", "part_name"), ("30", "asset_disposed_date"), ("53", "visit_end_datetime"), ("17", "skill_code"), ("35", "company_id"), ("45", "fault_description"), ("1", "company_id"), ("23", "asset_id"), ("18", "skill_description"), ("40", "skill_id"), ("62", "fault_status"), ("4", "company_address"), ("26", "asset_details"), ("59", "other_fault_details"), ("52", "visit_start_datetime"), ("16", "skill_id"), ("34", "engineer_id"), ("2", "company_type"), ("20", "staff_name"), ("39", "engineer_id"), ("46", "other_fault_details"), ("64", "skill_id"), ("5", "other_company_details"), ("58", "fault_description"), ("27", "asset_make"), ("41", "fault_log_entry_id"), ("63", "part_fault_id"), ("8", "contract_start_date"), ("55", "part_fault_id"), ("11", "part_id"), ("33", "part_id"), ("38", "other_details"), ("47", "engineer_visit_id"), ("3", "company_name"), ("21", "gender"), ("24", "maintenance_contract_id"), ("42", "asset_id"), ("60", "fault_log_entry_id"), ("6", "maintenance_contract_id"), ("9", "contract_end_date"), ("54", "other_visit_details"), ("10", "other_contract_details"), ("32", "asset_id")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_NUMBER), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_NUMBER), ("17", ColumnType_TEXT), ("18", ColumnType_TEXT), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_TEXT), ("21", ColumnType_TEXT), ("22", ColumnType_TEXT), ("23", ColumnType_NUMBER), ("24", ColumnType_NUMBER), ("25", ColumnType_NUMBER), ("26", ColumnType_TEXT), ("27", ColumnType_TEXT), ("28", ColumnType_TEXT), ("29", ColumnType_TIME), ("3", ColumnType_TEXT), ("30", ColumnType_TIME), ("31", ColumnType_TEXT), ("32", ColumnType_NUMBER), ("33", ColumnType_NUMBER), ("34", ColumnType_NUMBER), ("35", ColumnType_NUMBER), ("36", ColumnType_TEXT), ("37", ColumnType_TEXT), ("38", ColumnType_TEXT), ("39", ColumnType_NUMBER), ("4", ColumnType_TEXT), ("40", ColumnType_NUMBER), ("41", ColumnType_NUMBER), ("42", ColumnType_NUMBER), ("43", ColumnType_NUMBER), ("44", ColumnType_TIME), ("45", ColumnType_TEXT), ("46", ColumnType_TEXT), ("47", ColumnType_NUMBER), ("48", ColumnType_NUMBER), ("49", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("50", ColumnType_NUMBER), ("51", ColumnType_TEXT), ("52", ColumnType_TIME), ("53", ColumnType_TIME), ("54", ColumnType_TEXT), ("55", ColumnType_NUMBER), ("56", ColumnType_NUMBER), ("57", ColumnType_TEXT), ("58", ColumnType_TEXT), ("59", ColumnType_TEXT), ("6", ColumnType_NUMBER), ("60", ColumnType_NUMBER), ("61", ColumnType_NUMBER), ("62", ColumnType_TEXT), ("63", ColumnType_NUMBER), ("64", ColumnType_NUMBER), ("7", ColumnType_NUMBER), ("8", ColumnType_TIME), ("9", ColumnType_TIME)]
      tableNames = HashMap.fromList [("7", "Maintenance_Engineers"), ("13", "Skills_Required_To_Fix"), ("0", "Third_Party_Companies"), ("12", "Fault_Log_Parts"), ("1", "Maintenance_Contracts"), ("4", "Staff"), ("2", "Parts"), ("5", "Assets"), ("8", "Engineer_Skills"), ("11", "Part_Faults"), ("3", "Skills"), ("6", "Asset_Parts"), ("9", "Fault_Log"), ("10", "Engineer_Visits")]
      columnToTable = HashMap.fromList [("51", "10"), ("15", "2"), ("37", "7"), ("48", "10"), ("61", "12"), ("7", "1"), ("25", "5"), ("43", "9"), ("28", "5"), ("57", "11"), ("13", "2"), ("31", "5"), ("14", "2"), ("36", "7"), ("49", "10"), ("50", "10"), ("22", "4"), ("19", "4"), ("44", "9"), ("29", "5"), ("56", "11"), ("12", "2"), ("30", "5"), ("53", "10"), ("17", "3"), ("35", "7"), ("45", "9"), ("1", "0"), ("23", "5"), ("18", "3"), ("40", "8"), ("62", "12"), ("4", "0"), ("26", "5"), ("59", "11"), ("52", "10"), ("16", "3"), ("34", "7"), ("2", "0"), ("20", "4"), ("39", "8"), ("46", "9"), ("64", "13"), ("5", "0"), ("58", "11"), ("27", "5"), ("41", "9"), ("63", "13"), ("8", "1"), ("55", "11"), ("11", "2"), ("33", "6"), ("38", "7"), ("47", "10"), ("3", "0"), ("21", "4"), ("24", "5"), ("42", "9"), ("60", "12"), ("6", "1"), ("9", "1"), ("54", "10"), ("10", "1"), ("32", "6")]
      tableToColumns = HashMap.fromList [("7", ["34", "35", "36", "37", "38"]), ("13", ["63", "64"]), ("0", ["1", "2", "3", "4", "5"]), ("12", ["60", "61", "62"]), ("1", ["6", "7", "8", "9", "10"]), ("4", ["19", "20", "21", "22"]), ("2", ["11", "12", "13", "14", "15"]), ("5", ["23", "24", "25", "26", "27", "28", "29", "30", "31"]), ("8", ["39", "40"]), ("11", ["55", "56", "57", "58", "59"]), ("3", ["16", "17", "18"]), ("6", ["32", "33"]), ("9", ["41", "42", "43", "44", "45", "46"]), ("10", ["47", "48", "49", "50", "51", "52", "53", "54"])]
      foreignKeys = HashMap.fromList [("48", "19"), ("61", "55"), ("7", "1"), ("25", "1"), ("43", "19"), ("49", "34"), ("50", "41"), ("56", "11"), ("35", "1"), ("40", "16"), ("39", "34"), ("64", "16"), ("63", "55"), ("33", "11"), ("24", "6"), ("42", "23"), ("60", "41"), ("32", "23")]
      primaryKeys = ["1", "6", "11", "16", "19", "23", "34", "41", "47", "55"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

assetsMaintenanceQueries :: [Text.Text]
assetsMaintenanceQueries =
  []

-- "select * from ref_company_types",
-- "select t1.company_name from third_party_companies as t1 join maintenance_contracts as t2 on t1.company_id = t2.maintenance_contract_company_id join ref_company_types as t3 on t1.company_type_code = t3.company_type_code order by t2.contract_end_date desc limit 1"

assetsMaintenanceQueriesFails :: [Text.Text]
assetsMaintenanceQueriesFails = []

assetsMaintenanceParserTests :: TestItem
assetsMaintenanceParserTests =
  Group "assetsMaintenance" $
    (ParseQueryExprWithGuardsAndTypeChecking assetsMaintenanceSchema <$> assetsMaintenanceQueries)
      <> (ParseQueryExprWithGuards assetsMaintenanceSchema <$> assetsMaintenanceQueries)
      <> (ParseQueryExprWithoutGuards assetsMaintenanceSchema <$> assetsMaintenanceQueries)
      <> (ParseQueryExprFails assetsMaintenanceSchema <$> assetsMaintenanceQueriesFails)

assetsMaintenanceLexerTests :: TestItem
assetsMaintenanceLexerTests =
  Group "assetsMaintenance" $
    LexQueryExpr assetsMaintenanceSchema <$> assetsMaintenanceQueries
