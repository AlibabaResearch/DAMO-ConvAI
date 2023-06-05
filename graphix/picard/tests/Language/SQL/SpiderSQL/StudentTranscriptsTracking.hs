{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.StudentTranscriptsTracking where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

studentTranscriptsTrackingSchema :: SQLSchema
studentTranscriptsTrackingSchema =
  let columnNames = HashMap.fromList [("1", "address_id"), ("10", "course_id"), ("11", "course_name"), ("12", "course_description"), ("13", "other_details"), ("14", "department_id"), ("15", "department_name"), ("16", "department_description"), ("17", "other_details"), ("18", "degree_program_id"), ("19", "department_id"), ("2", "line_1"), ("20", "degree_summary_name"), ("21", "degree_summary_description"), ("22", "other_details"), ("23", "section_id"), ("24", "course_id"), ("25", "section_name"), ("26", "section_description"), ("27", "other_details"), ("28", "semester_id"), ("29", "semester_name"), ("3", "line_2"), ("30", "semester_description"), ("31", "other_details"), ("32", "student_id"), ("33", "current_address_id"), ("34", "permanent_address_id"), ("35", "first_name"), ("36", "middle_name"), ("37", "last_name"), ("38", "cell_mobile_number"), ("39", "email_address"), ("4", "line_3"), ("40", "ssn"), ("41", "date_first_registered"), ("42", "date_left"), ("43", "other_student_details"), ("44", "student_enrolment_id"), ("45", "degree_program_id"), ("46", "semester_id"), ("47", "student_id"), ("48", "other_details"), ("49", "student_course_id"), ("5", "city"), ("50", "course_id"), ("51", "student_enrolment_id"), ("52", "transcript_id"), ("53", "transcript_date"), ("54", "other_details"), ("55", "student_course_id"), ("56", "transcript_id"), ("6", "zip_postcode"), ("7", "state_province_county"), ("8", "country"), ("9", "other_address_details")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_TEXT), ("12", ColumnType_TEXT), ("13", ColumnType_TEXT), ("14", ColumnType_NUMBER), ("15", ColumnType_TEXT), ("16", ColumnType_TEXT), ("17", ColumnType_TEXT), ("18", ColumnType_NUMBER), ("19", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("20", ColumnType_TEXT), ("21", ColumnType_TEXT), ("22", ColumnType_TEXT), ("23", ColumnType_NUMBER), ("24", ColumnType_NUMBER), ("25", ColumnType_TEXT), ("26", ColumnType_TEXT), ("27", ColumnType_TEXT), ("28", ColumnType_NUMBER), ("29", ColumnType_TEXT), ("3", ColumnType_TEXT), ("30", ColumnType_TEXT), ("31", ColumnType_TEXT), ("32", ColumnType_NUMBER), ("33", ColumnType_NUMBER), ("34", ColumnType_NUMBER), ("35", ColumnType_TEXT), ("36", ColumnType_TEXT), ("37", ColumnType_TEXT), ("38", ColumnType_TEXT), ("39", ColumnType_TEXT), ("4", ColumnType_TEXT), ("40", ColumnType_TEXT), ("41", ColumnType_TIME), ("42", ColumnType_TIME), ("43", ColumnType_TEXT), ("44", ColumnType_NUMBER), ("45", ColumnType_NUMBER), ("46", ColumnType_NUMBER), ("47", ColumnType_NUMBER), ("48", ColumnType_TEXT), ("49", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("50", ColumnType_NUMBER), ("51", ColumnType_NUMBER), ("52", ColumnType_NUMBER), ("53", ColumnType_TIME), ("54", ColumnType_TEXT), ("55", ColumnType_NUMBER), ("56", ColumnType_NUMBER), ("6", ColumnType_TEXT), ("7", ColumnType_TEXT), ("8", ColumnType_TEXT), ("9", ColumnType_TEXT)]
      tableNames = HashMap.fromList [("0", "Addresses"), ("1", "Courses"), ("10", "Transcript_Contents"), ("2", "Departments"), ("3", "Degree_Programs"), ("4", "Sections"), ("5", "Semesters"), ("6", "Students"), ("7", "Student_Enrolment"), ("8", "Student_Enrolment_Courses"), ("9", "Transcripts")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "1"), ("12", "1"), ("13", "1"), ("14", "2"), ("15", "2"), ("16", "2"), ("17", "2"), ("18", "3"), ("19", "3"), ("2", "0"), ("20", "3"), ("21", "3"), ("22", "3"), ("23", "4"), ("24", "4"), ("25", "4"), ("26", "4"), ("27", "4"), ("28", "5"), ("29", "5"), ("3", "0"), ("30", "5"), ("31", "5"), ("32", "6"), ("33", "6"), ("34", "6"), ("35", "6"), ("36", "6"), ("37", "6"), ("38", "6"), ("39", "6"), ("4", "0"), ("40", "6"), ("41", "6"), ("42", "6"), ("43", "6"), ("44", "7"), ("45", "7"), ("46", "7"), ("47", "7"), ("48", "7"), ("49", "8"), ("5", "0"), ("50", "8"), ("51", "8"), ("52", "9"), ("53", "9"), ("54", "9"), ("55", "10"), ("56", "10"), ("6", "0"), ("7", "0"), ("8", "0"), ("9", "0")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7", "8", "9"]), ("1", ["10", "11", "12", "13"]), ("10", ["55", "56"]), ("2", ["14", "15", "16", "17"]), ("3", ["18", "19", "20", "21", "22"]), ("4", ["23", "24", "25", "26", "27"]), ("5", ["28", "29", "30", "31"]), ("6", ["32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"]), ("7", ["44", "45", "46", "47", "48"]), ("8", ["49", "50", "51"]), ("9", ["52", "53", "54"])]
      foreignKeys = HashMap.fromList [("19", "14"), ("24", "10"), ("33", "1"), ("34", "1"), ("45", "18"), ("46", "28"), ("47", "32"), ("50", "10"), ("51", "44"), ("55", "49"), ("56", "52")]
      primaryKeys = ["1", "10", "14", "18", "23", "28", "32", "44", "49", "52"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

studentTranscriptsTrackingQueries :: [Text.Text]
studentTranscriptsTrackingQueries =
  [ "select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id = t2.address_id where t2.country = 'haiti' or t1.cell_mobile_number = '09700166582'"
  ]

studentTranscriptsTrackingQueriesFails :: [Text.Text]
studentTranscriptsTrackingQueriesFails = []

studentTranscriptsTrackingParserTests :: TestItem
studentTranscriptsTrackingParserTests =
  Group "studentTranscriptsTracking" $
    (ParseQueryExprWithGuardsAndTypeChecking studentTranscriptsTrackingSchema <$> studentTranscriptsTrackingQueries)
      <> (ParseQueryExprWithGuards studentTranscriptsTrackingSchema <$> studentTranscriptsTrackingQueries)
      <> (ParseQueryExprWithoutGuards studentTranscriptsTrackingSchema <$> studentTranscriptsTrackingQueries)
      <> (ParseQueryExprFails studentTranscriptsTrackingSchema <$> studentTranscriptsTrackingQueriesFails)

studentTranscriptsTrackingLexerTests :: TestItem
studentTranscriptsTrackingLexerTests =
  Group "studentTranscriptsTracking" $
    LexQueryExpr studentTranscriptsTrackingSchema <$> studentTranscriptsTrackingQueries
