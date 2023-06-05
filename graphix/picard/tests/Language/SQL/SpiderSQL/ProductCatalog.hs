{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.ProductCatalog where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

productCatalogSchema :: SQLSchema
productCatalogSchema =
  let columnNames = HashMap.fromList [("1", "attribute_id"), ("10", "catalog_id"), ("11", "catalog_level_name"), ("12", "catalog_entry_id"), ("13", "catalog_level_number"), ("14", "parent_entry_id"), ("15", "previous_entry_id"), ("16", "next_entry_id"), ("17", "catalog_entry_name"), ("18", "product_stock_number"), ("19", "price_in_dollars"), ("2", "attribute_name"), ("20", "price_in_euros"), ("21", "price_in_pounds"), ("22", "capacity"), ("23", "length"), ("24", "height"), ("25", "width"), ("26", "catalog_entry_id"), ("27", "catalog_level_number"), ("28", "attribute_id"), ("29", "attribute_value"), ("3", "attribute_data_type"), ("4", "catalog_id"), ("5", "catalog_name"), ("6", "catalog_publisher"), ("7", "date_of_publication"), ("8", "date_of_latest_revision"), ("9", "catalog_level_number")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_NUMBER), ("13", ColumnType_NUMBER), ("14", ColumnType_NUMBER), ("15", ColumnType_NUMBER), ("16", ColumnType_NUMBER), ("17", ColumnType_TEXT), ("18", ColumnType_TEXT), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_NUMBER), ("21", ColumnType_NUMBER), ("22", ColumnType_TEXT), ("23", ColumnType_TEXT), ("24", ColumnType_TEXT), ("25", ColumnType_TEXT), ("26", ColumnType_NUMBER), ("27", ColumnType_NUMBER), ("28", ColumnType_NUMBER), ("29", ColumnType_TEXT), ("3", ColumnType_TEXT), ("4", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("6", ColumnType_TEXT), ("7", ColumnType_TIME), ("8", ColumnType_TIME), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "Attribute_Definitions"), ("1", "Catalogs"), ("2", "Catalog_Structure"), ("3", "Catalog_Contents"), ("4", "Catalog_Contents_Additional_Attributes")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "2"), ("11", "2"), ("12", "3"), ("13", "3"), ("14", "3"), ("15", "3"), ("16", "3"), ("17", "3"), ("18", "3"), ("19", "3"), ("2", "0"), ("20", "3"), ("21", "3"), ("22", "3"), ("23", "3"), ("24", "3"), ("25", "3"), ("26", "4"), ("27", "4"), ("28", "4"), ("29", "4"), ("3", "0"), ("4", "1"), ("5", "1"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3"]), ("1", ["4", "5", "6", "7", "8"]), ("2", ["9", "10", "11"]), ("3", ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]), ("4", ["26", "27", "28", "29"])]
      foreignKeys = HashMap.fromList [("10", "4"), ("13", "9"), ("26", "12"), ("27", "9")]
      primaryKeys = ["1", "4", "9", "12"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

productCatalogQueries :: [Text.Text]
productCatalogQueries =
  [ "select catalog_entry_name from catalog_contents as T1",
    "select distinct (T1.catalog_entry_name) from catalog_contents as T1",
    "select distinct(catalog_entry_name) from catalog_contents",
    "select distinct(catalog_publisher) from catalogs where catalog_publisher like \"%murray%\""
  ]

productCatalogQueriesFails :: [Text.Text]
productCatalogQueriesFails = []

productCatalogParserTests :: TestItem
productCatalogParserTests =
  Group "productCatalog" $
    (ParseQueryExprWithGuardsAndTypeChecking productCatalogSchema <$> productCatalogQueries)
      <> (ParseQueryExprWithGuards productCatalogSchema <$> productCatalogQueries)
      <> (ParseQueryExprWithoutGuards productCatalogSchema <$> productCatalogQueries)
      <> (ParseQueryExprFails productCatalogSchema <$> productCatalogQueriesFails)

productCatalogLexerTests :: TestItem
productCatalogLexerTests =
  Group "productCatalog" $
    LexQueryExpr productCatalogSchema <$> productCatalogQueries
