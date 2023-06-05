{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Chinook1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

chinook1Schema :: SQLSchema
chinook1Schema =
  let columnNames = HashMap.fromList [("1", "AlbumId"), ("10", "Address"), ("11", "City"), ("12", "State"), ("13", "Country"), ("14", "PostalCode"), ("15", "Phone"), ("16", "Fax"), ("17", "Email"), ("18", "SupportRepId"), ("19", "EmployeeId"), ("2", "Title"), ("20", "LastName"), ("21", "FirstName"), ("22", "Title"), ("23", "ReportsTo"), ("24", "BirthDate"), ("25", "HireDate"), ("26", "Address"), ("27", "City"), ("28", "State"), ("29", "Country"), ("3", "ArtistId"), ("30", "PostalCode"), ("31", "Phone"), ("32", "Fax"), ("33", "Email"), ("34", "GenreId"), ("35", "Name"), ("36", "InvoiceId"), ("37", "CustomerId"), ("38", "InvoiceDate"), ("39", "BillingAddress"), ("4", "ArtistId"), ("40", "BillingCity"), ("41", "BillingState"), ("42", "BillingCountry"), ("43", "BillingPostalCode"), ("44", "Total"), ("45", "InvoiceLineId"), ("46", "InvoiceId"), ("47", "TrackId"), ("48", "UnitPrice"), ("49", "Quantity"), ("5", "Name"), ("50", "MediaTypeId"), ("51", "Name"), ("52", "PlaylistId"), ("53", "Name"), ("54", "PlaylistId"), ("55", "TrackId"), ("56", "TrackId"), ("57", "Name"), ("58", "AlbumId"), ("59", "MediaTypeId"), ("6", "CustomerId"), ("60", "GenreId"), ("61", "Composer"), ("62", "Milliseconds"), ("63", "Bytes"), ("64", "UnitPrice"), ("7", "FirstName"), ("8", "LastName"), ("9", "Company")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_TEXT), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_TEXT), ("15", ColumnType_TEXT), ("16", ColumnType_TEXT), ("17", ColumnType_TEXT), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_TEXT), ("21", ColumnType_TEXT), ("22", ColumnType_TEXT), ("23", ColumnType_NUMBER), ("24", ColumnType_TIME), ("25", ColumnType_TIME), ("26", ColumnType_TEXT), ("27", ColumnType_TEXT), ("28", ColumnType_TEXT), ("29", ColumnType_TEXT), ("3", ColumnType_NUMBER), ("30", ColumnType_TEXT), ("31", ColumnType_TEXT), ("32", ColumnType_TEXT), ("33", ColumnType_TEXT), ("34", ColumnType_NUMBER), ("35", ColumnType_TEXT), ("36", ColumnType_NUMBER), ("37", ColumnType_NUMBER), ("38", ColumnType_TIME), ("39", ColumnType_TEXT), ("4", ColumnType_NUMBER), ("40", ColumnType_TEXT), ("41", ColumnType_TEXT), ("42", ColumnType_TEXT), ("43", ColumnType_TEXT), ("44", ColumnType_NUMBER), ("45", ColumnType_NUMBER), ("46", ColumnType_NUMBER), ("47", ColumnType_NUMBER), ("48", ColumnType_NUMBER), ("49", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("50", ColumnType_NUMBER), ("51", ColumnType_TEXT), ("52", ColumnType_NUMBER), ("53", ColumnType_TEXT), ("54", ColumnType_NUMBER), ("55", ColumnType_NUMBER), ("56", ColumnType_NUMBER), ("57", ColumnType_TEXT), ("58", ColumnType_NUMBER), ("59", ColumnType_NUMBER), ("6", ColumnType_NUMBER), ("60", ColumnType_NUMBER), ("61", ColumnType_TEXT), ("62", ColumnType_NUMBER), ("63", ColumnType_NUMBER), ("64", ColumnType_NUMBER), ("7", ColumnType_TEXT), ("8", ColumnType_TEXT), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "Album"), ("1", "Artist"), ("10", "Track"), ("2", "Customer"), ("3", "Employee"), ("4", "Genre"), ("5", "Invoice"), ("6", "InvoiceLine"), ("7", "MediaType"), ("8", "Playlist"), ("9", "PlaylistTrack")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "2"), ("11", "2"), ("12", "2"), ("13", "2"), ("14", "2"), ("15", "2"), ("16", "2"), ("17", "2"), ("18", "2"), ("19", "3"), ("2", "0"), ("20", "3"), ("21", "3"), ("22", "3"), ("23", "3"), ("24", "3"), ("25", "3"), ("26", "3"), ("27", "3"), ("28", "3"), ("29", "3"), ("3", "0"), ("30", "3"), ("31", "3"), ("32", "3"), ("33", "3"), ("34", "4"), ("35", "4"), ("36", "5"), ("37", "5"), ("38", "5"), ("39", "5"), ("4", "1"), ("40", "5"), ("41", "5"), ("42", "5"), ("43", "5"), ("44", "5"), ("45", "6"), ("46", "6"), ("47", "6"), ("48", "6"), ("49", "6"), ("5", "1"), ("50", "7"), ("51", "7"), ("52", "8"), ("53", "8"), ("54", "9"), ("55", "9"), ("56", "10"), ("57", "10"), ("58", "10"), ("59", "10"), ("6", "2"), ("60", "10"), ("61", "10"), ("62", "10"), ("63", "10"), ("64", "10"), ("7", "2"), ("8", "2"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3"]), ("1", ["4", "5"]), ("10", ["56", "57", "58", "59", "60", "61", "62", "63", "64"]), ("2", ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]), ("3", ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]), ("4", ["34", "35"]), ("5", ["36", "37", "38", "39", "40", "41", "42", "43", "44"]), ("6", ["45", "46", "47", "48", "49"]), ("7", ["50", "51"]), ("8", ["52", "53"]), ("9", ["54", "55"])]
      foreignKeys = HashMap.fromList [("18", "19"), ("23", "19"), ("3", "4"), ("37", "6"), ("46", "36"), ("47", "56"), ("54", "52"), ("55", "56"), ("58", "1"), ("59", "50"), ("60", "34")]
      primaryKeys = ["1", "4", "6", "19", "34", "36", "45", "50", "52", "54", "56"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

chinook1Queries :: [Text.Text]
chinook1Queries =
  [ "select distinct(billingcountry) from invoice",
    "select t2.name, t1.artistid from album as t1 join artist as t2 on t1.artistid = t2.artistid group by t1.artistid having count(*) >= 3 order by t2.name",
    "select distinct(unitprice) from track"
  ]

chinook1QueriesFails :: [Text.Text]
chinook1QueriesFails = []

chinook1ParserTests :: TestItem
chinook1ParserTests =
  Group "chinook1" $
    (ParseQueryExprWithGuardsAndTypeChecking chinook1Schema <$> chinook1Queries)
      <> (ParseQueryExprWithGuards chinook1Schema <$> chinook1Queries)
      <> (ParseQueryExprWithoutGuards chinook1Schema <$> chinook1Queries)
      <> (ParseQueryExprFails chinook1Schema <$> chinook1QueriesFails)

chinook1LexerTests :: TestItem
chinook1LexerTests =
  Group "chinook1" $
    LexQueryExpr chinook1Schema <$> chinook1Queries
