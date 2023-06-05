{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Pets1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (ColumnType (..), SQLSchema (..))

pets1Schema :: SQLSchema
pets1Schema =
  let columnNames = HashMap.fromList [("1", "StuID"), ("10", "PetID"), ("11", "PetID"), ("12", "PetType"), ("13", "pet_age"), ("14", "weight"), ("2", "LName"), ("3", "Fname"), ("4", "Age"), ("5", "Sex"), ("6", "Major"), ("7", "Advisor"), ("8", "city_code"), ("9", "StuID")]
      columnTypes = HashMap.fromList [("1", ColumnType_NUMBER), ("10", ColumnType_NUMBER), ("11", ColumnType_NUMBER), ("12", ColumnType_TEXT), ("13", ColumnType_NUMBER), ("14", ColumnType_NUMBER), ("2", ColumnType_TEXT), ("3", ColumnType_TEXT), ("4", ColumnType_NUMBER), ("5", ColumnType_TEXT), ("6", ColumnType_NUMBER), ("7", ColumnType_NUMBER), ("8", ColumnType_TEXT), ("9", ColumnType_NUMBER)]
      tableNames = HashMap.fromList [("0", "Student"), ("1", "Has_Pet"), ("2", "Pets")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "1"), ("11", "2"), ("12", "2"), ("13", "2"), ("14", "2"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "0"), ("7", "0"), ("8", "0"), ("9", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7", "8"]), ("1", ["9", "10"]), ("2", ["11", "12", "13", "14"])]
      foreignKeys = HashMap.fromList [("10", "11"), ("9", "1")]
      primaryKeys = ["1", "11"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}

pets1Queries :: [Text.Text]
pets1Queries =
  [ "SELECT count(*) FROM pets WHERE weight  >  10",
    "SELECT weight FROM pets ORDER BY pet_age LIMIT 1",
    "SELECT max(weight) ,  petType FROM pets GROUP BY petType",
    "SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.age  >  20",
    "SELECT count(*) FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid WHERE T1.sex  =  'F' AND T3.pettype  =  'dog'",
    "SELECT count(DISTINCT pettype) FROM pets",
    "SELECT DISTINCT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' OR T3.pettype  =  'dog'",
    "select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'",
    "SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat' INTERSECT SELECT T1.Fname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog'",
    "SELECT major ,  age FROM student WHERE stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')",
    "SELECT stuid FROM student EXCEPT SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat'",
    "SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')",
    "SELECT pettype ,  weight FROM pets ORDER BY pet_age LIMIT 1",
    "SELECT petid ,  weight FROM pets WHERE pet_age  >  1",
    "SELECT avg(pet_age) ,  max(pet_age) ,  pettype FROM pets GROUP BY pettype",
    "SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype",
    "SELECT DISTINCT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid",
    "SELECT T2.petid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid WHERE T1.Lname  =  'Smith'",
    "SELECT count(*) ,  T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid GROUP BY T1.stuid",
    "select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid",
    "SELECT T1.fname ,  T1.sex FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid GROUP BY T1.stuid HAVING count(*)  >  1",
    "SELECT T1.lname FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pet_age  =  3 AND T3.pettype  =  'cat'",
    "select avg(age) from student where stuid not in (select stuid from has_pet)"
  ]

pets1ParserTests :: TestItem
pets1ParserTests =
  Group "pets_1" $
    (ParseQueryExprWithGuardsAndTypeChecking pets1Schema <$> pets1Queries)
      <> (ParseQueryExprWithGuards pets1Schema <$> pets1Queries)
      <> (ParseQueryExprWithoutGuards pets1Schema <$> pets1Queries)
      <> (ParseQueryExprFails pets1Schema <$> [])

pets1LexerTests :: TestItem
pets1LexerTests =
  Group "pets_1" $ LexQueryExpr pets1Schema <$> pets1Queries
