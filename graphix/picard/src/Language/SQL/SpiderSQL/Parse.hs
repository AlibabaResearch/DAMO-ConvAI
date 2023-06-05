{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE StandaloneDeriving #-}

module Language.SQL.SpiderSQL.Parse where

import Control.Applicative (Alternative (..), Applicative (liftA2), optional)
import Control.Lens ((%~), (^.))
import Control.Monad (MonadPlus, forM_, join, unless, when)
import Control.Monad.Reader (runReaderT)
import Control.Monad.Reader.Class (MonadReader (ask))
import Control.Monad.State.Class (MonadState (..), modify)
import Control.Monad.State.Strict (evalStateT)
import Data.Char (isAlpha, isAlphaNum, toLower)
import Data.Foldable (Foldable (foldl'), for_)
import Data.Functor (($>))
import Data.Generics.Product (field)
import qualified Data.HashMap.Strict as HashMap (HashMap, compose, filter, insertWith, intersectionWith, keys, lookup, member, toList)
import qualified Data.HashSet as HashSet
import Data.Hashable (Hashable)
import qualified Data.Map as Map (Map, fromListWith, lookupLE, member, singleton, toList, union, unionWith)
import Data.Maybe (catMaybes, fromMaybe)
import qualified Data.Text as Text
import Data.Word (Word8)
import GHC.Generics (Generic)
import Language.SQL.SpiderSQL.Prelude (columnNameP, columnTypeAndNameP, doubleP, eitherP, intP, isAnd, isAs, isAsc, isAvg, isBetween, isClosedParenthesis, isComma, isCount, isDesc, isDistinct, isDivide, isDot, isEq, isExcept, isFrom, isGe, isGroupBy, isGt, isHaving, isIn, isIntersect, isJoin, isLe, isLike, isLimit, isLt, isMax, isMin, isMinus, isNe, isNot, isOn, isOpenParenthesis, isOr, isOrderBy, isPlus, isSelect, isStar, isSum, isTimes, isUnion, isWhere, manyAtMost, quotedString, tableNameP, toColumnType)
import Language.SQL.SpiderSQL.Syntax (Agg (..), AggType (..), Alias (..), ColUnit (..), ColumnId (..), Cond (..), From (..), OrderBy (..), OrderByOrder (..), SX (..), Select (..), SpiderSQL (..), SpiderTyp (..), TableId (..), TableUnit (..), Val (..), ValTC, ValUnit (..), X (..), XValUnit, colUnitTyp, selectTyp, spiderSQLTyp, valUnitTyp, pattern AggUD, pattern ColUnitUD, pattern ColumnIdUD, pattern ColumnUD, pattern DistinctColUnitUD, pattern SelectDistinctUD, pattern SelectUD, pattern SpiderSQLUD)
import qualified Picard.Types (ColumnId, SQLSchema (..))
import Text.Parser.Char (CharParsing (..), alphaNum, digit, spaces)
import Text.Parser.Combinators (Parsing (..), between, choice, sepBy, sepBy1)
import Text.Parser.Permutation (permute, (<$$>), (<||>))
import Text.Parser.Token (TokenParsing (..))

-- $setup
-- >>> :set -XOverloadedStrings
-- >>> import qualified Data.Attoparsec.Text as Atto (parse, parseOnly, endOfInput, string, char)
-- >>> import Picard.Types (SQLSchema (..))
-- >>> import Control.Monad.Reader (runReader, runReaderT)
-- >>> import Control.Monad.Trans (MonadTrans (lift))
-- >>> import qualified Data.HashMap.Strict as HashMap
-- >>> columnNames = HashMap.fromList [("1", "Singer_ID"), ("2", "Name"), ("3", "Birth_Year"), ("4", "Net_Worth_Millions"), ("5", "Citizenship"), ("6", "Song_ID"), ("7", "Title"), ("8", "Singer_ID"), ("9", "Sales"), ("10", "Highest_Position")] :: HashMap.HashMap Text.Text Text.Text
-- >>> columnTypes = HashMap.fromList []
-- >>> tableNames = HashMap.fromList [("0", "singer"), ("1", "song")] :: HashMap.HashMap Text.Text Text.Text
-- >>> columnToTable = HashMap.fromList [("1", "0"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "1"), ("10", "1")] :: HashMap.HashMap Text.Text Text.Text
-- >>> tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5"]), ("1", ["6", "7", "8", "9", "10"])] :: HashMap.HashMap Text.Text [Text.Text]
-- >>> foreignKeys = HashMap.fromList [("8", "1")] :: HashMap.HashMap Text.Text Text.Text
-- >>> primaryKeys = ["1", "6"] :: [Text.Text]
-- >>> sqlSchema = SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}
-- >>> parserEnv = ParserEnv (ParserEnvWithGuards (withGuards SUD)) sqlSchema
-- >>> testParse = Atto.parse . flip runReaderT parserEnv . flip evalStateT mkParserStateUD
-- >>> testParseOnly p = (Atto.parseOnly . flip runReaderT parserEnv . flip evalStateT mkParserStateUD) (p <* (lift . lift) Atto.endOfInput)
-- >>> spiderSQLTestParse = (Atto.parse . flip runReaderT parserEnv) (spiderSQL SUD mkParserStateUD)
-- >>> spiderSQLTestParseOnly = (Atto.parseOnly . flip runReaderT parserEnv) (spiderSQL SUD mkParserStateUD <* lift Atto.endOfInput)

-- | ParserState
--
-- A table alias defined in scope n is:
-- - valid in scopes n, n + 1, ... unless shadowed by an alias defined in scope n' > n,
-- - not valid in scopes 0, 1, ..., n - 1.
data ParserState x = ParserState
  { psAliases :: HashMap.HashMap Alias (Map.Map Scope (TableUnit x)),
    psTables :: HashMap.HashMap (Either TableId (Select x)) (HashSet.HashSet Scope),
    psCurScope :: Scope,
    psGuards :: HashMap.HashMap Scope (HashSet.HashSet (Guard x))
  }
  deriving stock (Generic)

type ParserStateUD = ParserState 'UD

deriving stock instance Eq ParserStateUD

deriving stock instance Show ParserStateUD

newtype Scope = Scope Word8
  deriving stock (Generic)
  deriving (Show, Eq, Ord, Num, Enum, Bounded, Hashable) via Word8

data Guard x
  = GuardTableColumn TableId (ColumnId x)
  | GuardAliasColumn Alias (ColumnId x)
  | GuardColumn (ColumnId x)
  deriving stock (Generic)

type GuardUD = Guard 'UD

deriving stock instance Eq GuardUD

deriving stock instance Show GuardUD

deriving anyclass instance Hashable GuardUD

type GuardTC = Guard 'TC

deriving stock instance Eq GuardTC

deriving stock instance Show GuardTC

deriving anyclass instance Hashable GuardTC

mkParserStateUD :: ParserState 'UD
mkParserStateUD = ParserState {psAliases = mempty, psTables = mempty, psCurScope = minBound, psGuards = mempty}

mkParserStateTC :: ParserState 'TC
mkParserStateTC = ParserState {psAliases = mempty, psTables = mempty, psCurScope = minBound, psGuards = mempty}

newtype ParserEnvWithGuards x
  = ParserEnvWithGuards
      ( forall m p.
        ( Parsing m,
          MonadPlus m,
          MonadState (ParserState x) m
        ) =>
        Picard.Types.SQLSchema ->
        m p ->
        m p
      )

-- | ParserEnv
data ParserEnv x = ParserEnv
  { peWithGuards :: ParserEnvWithGuards x,
    peSQLSchema :: Picard.Types.SQLSchema
  }
  deriving stock (Generic)

type MonadSQL x m = (MonadPlus m, MonadState (ParserState x) m, MonadReader (ParserEnv x) m)

-- >>> testParseOnly (betweenParentheses $ char 'x') "x"
-- Left "\"(\": satisfyElem"
--
-- >>> testParseOnly (betweenParentheses $ char 'x') "(x)"
-- Right 'x'
--
-- >>> testParseOnly (betweenParentheses $ char 'x') "( x )"
-- Right 'x'
betweenParentheses :: CharParsing m => m a -> m a
betweenParentheses =
  between
    (try $ isOpenParenthesis <* spaces)
    (try $ spaces *> isClosedParenthesis)

-- >>> testParseOnly (betweenOptionalParentheses $ char 'x') "x"
-- Right 'x'
--
-- >>> testParseOnly (betweenOptionalParentheses $ char 'x') "(x)"
-- Right 'x'
betweenOptionalParentheses :: CharParsing m => m a -> m a
betweenOptionalParentheses p = try (betweenParentheses p) <|> try p

-- | 'Select' parser
--
-- >>> testParseOnly (select SUD) "select *"
-- Right (Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))])
--
-- >>> testParseOnly (select SUD) "select count singer.*"
-- Right (Select () [Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = Star}}))])
--
-- >>> testParseOnly (select SUD) "SELECT COUNT (DISTINCT song.Title)"
-- Right (Select () [Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Nothing, distinctColUnitTable = Just (Left (TableId {tableName = "song"})), distinctColUnitColdId = ColumnId {columnIdX = (), columnName = "Title"}}}))])
--
-- >>> testParseOnly (select SUD) "SELECT COUNT (DISTINCT T1.Title)"
-- Right (Select () [Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Nothing, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = ColumnId {columnIdX = (), columnName = "Title"}}}))])
select :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (Select x)
select sx = flip (<?>) "select" $ do
  _ <- isSelect
  someSpace
  distinct <- optional (try $ isDistinct <* spaces)
  aggs <- sepBy (try $ betweenOptionalParentheses (agg sx)) (try $ spaces *> isComma <* spaces)
  case sx of
    SUD -> case distinct of
      Just _ -> pure $ SelectDistinctUD aggs
      Nothing -> pure $ SelectUD aggs
    STC ->
      let typ = (\case Agg typ' _ _ -> typ') <$> aggs
       in case distinct of
            Just _ -> pure $ SelectDistinct typ aggs
            Nothing -> pure $ Select typ aggs

-- | 'Agg' parser.
--
-- >>> testParseOnly (agg SUD) "singer.Singer_ID"
-- Right (Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly (agg SUD) "count *"
-- Right (Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly (agg SUD) "count (*)"
-- Right (Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly (agg SUD) "count(*)"
-- Right (Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly (agg SUD) "count singer.Singer_ID"
-- Right (Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
agg :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (Agg x)
agg sx =
  flip (<?>) "agg" $ do
    at <- optional . try $ aggType
    vu <- case at of
      Nothing -> valUnit sx
      Just _ ->
        try (spaces *> betweenParentheses (spaces *> valUnit sx <* spaces))
          <|> try (someSpace *> valUnit sx)
    case sx of
      SUD -> pure $ AggUD at vu
      STC -> case (vu, at) of
        (Column typ _, Nothing) -> pure $ Agg typ at vu
        (Column _ _, Just Count) -> pure $ Agg TNumber at vu
        (Column TNumber _, Just _) -> pure $ Agg TNumber at vu
        (Column _ val', Just at') ->
          unexpected $
            "value " <> show val' <> " does not support " <> show at' <> " aggregation"
        (_, _) -> pure $ Agg TNumber at vu

-- | 'AggType' parser.
--
-- >>> testParseOnly aggType ""
-- Left "aggType: Failed reading: mzero"
--
-- >>> testParseOnly aggType "sum"
-- Right Sum
aggType :: forall m. CharParsing m => m AggType
aggType = flip (<?>) "aggType" $ choice choices
  where
    choices =
      [ try $ isMax $> Max,
        try $ isMin $> Min,
        try $ isCount $> Count,
        try $ isSum $> Sum,
        try $ isAvg $> Avg
      ]

-- | 'ValUnit' parser.
--
-- >>> testParseOnly (valUnit SUD) "t1.Singer_ID"
-- Right (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}))
--
-- >>> testParseOnly (valUnit SUD) "t2.Sales / t1.Net_Worth_Millions"
-- Right (Divide () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Sales"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Net_Worth_Millions"}}}))
--
-- >>> testParseOnly (valUnit SUD) "t2.Sales / 4"
-- Right (Divide () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Sales"}}}) (Number {numberValue = 4.0}))
valUnit :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (ValUnit x)
valUnit sx =
  flip (<?>) "valUnit" $ do
    column <- val sx
    maybeBinary <- do
      let vTyp :: ValTC -> m SpiderTyp
          vTyp (ValColUnit typ _) = pure typ
          vTyp (Number _) = pure TNumber
          vTyp v@(ValString _) = unexpected $ "unexpected " <> show v
          vTyp (ValSQL [typ] _) = pure typ
          vTyp v@(ValSQL _ _) = unexpected $ "unexpected " <> show v
          unifyVTyps vt vt'
            | vt == TStar && vt' == TStar = pure TStar
            | vt == TStar && vt' == TNumber = pure TStar
            | vt == TNumber && vt' == TStar = pure TStar
            | vt == TNumber && vt' == TNumber = pure TNumber
            | vt == TStar && vt' == TTime = pure TStar
            | vt == TTime && vt' == TStar = pure TStar
            | vt == TTime && vt' == TTime = pure TTime
            | otherwise = unexpected $ "the types " <> show vt <> " and " <> show vt' <> " are incompatible"
          checkedBinary :: SX x -> (XValUnit x -> Val x -> Val x -> ValUnit x) -> Val x -> Val x -> m (ValUnit x)
          checkedBinary SUD f v v' = pure $ f () v v'
          checkedBinary STC f v v' = do
            vt <- vTyp v
            vt' <- vTyp v'
            vt'' <- unifyVTyps vt vt'
            pure $ f vt'' v v'
      optional
        ( try $
            someSpace
              *> choice
                [ try $ isMinus $> checkedBinary sx Minus column,
                  try $ isPlus $> checkedBinary sx Plus column,
                  try $ isTimes $> checkedBinary sx Times column,
                  try $ isDivide $> checkedBinary sx Divide column
                ]
        )
    case sx of
      SUD -> case maybeBinary of
        Nothing -> pure $ ColumnUD column
        Just binary -> do
          v' <- someSpace *> val sx
          binary v'
      STC -> case maybeBinary of
        Nothing -> case column of
          ValColUnit typ _ -> pure $ Column typ column
          Number _ -> pure $ Column TNumber column
          ValString _ -> pure $ Column TText column
          ValSQL [typ] _ -> pure $ Column typ column
          ValSQL _ _ -> unexpected $ "unexpected " <> show column
        Just binary -> do
          v' <- someSpace *> val sx
          binary v'

-- | 'ColUnit' parser.
--
-- >>> testParseOnly (colUnit SUD) "*"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly (colUnit SUD) "Singer_ID"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}})
--
-- >>> testParseOnly (colUnit SUD) "distinct Singer_ID"
-- Right (DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Nothing, distinctColUnitTable = Nothing, distinctColUnitColdId = ColumnId {columnIdX = (), columnName = "Singer_ID"}})
--
-- >>> testParseOnly (colUnit SUD) "t1.Singer_ID"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}})
--
-- >>> testParseOnly (colUnit SUD) "count *"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count (*)"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count(*)"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count ( * )"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count t1.Singer_ID"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}})
--
-- >>> testParseOnly (colUnit SUD) "count distinct t1.*"
-- Right (DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count (distinct t1.*)"
-- Right (DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> testParseOnly (colUnit SUD) "count(distinct t1.*)"
-- Right (DistinctColUnit {distinctColUnitX = (), distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> (Atto.parseOnly . flip runReaderT (ParserEnv (ParserEnvWithGuards (withGuards SUD)) sqlSchema { sQLSchema_columnNames = HashMap.union (sQLSchema_columnNames sqlSchema) (HashMap.singleton "11" "country") }) . flip evalStateT mkParserStateUD) (colUnit SUD) "country"
-- Right (ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "country"}})
colUnit ::
  forall x m.
  ( TokenParsing m,
    MonadSQL x m
  ) =>
  SX x ->
  m (ColUnit x)
colUnit sx = flip (<?>) "colUnit" $ do
  at <- optional . try $ aggType
  (distinct, tabAli, col) <- do
    let p = do
          distinct <- optional (try $ isDistinct <* someSpace)
          (tabAli, col) <-
            try ((Nothing,) <$> columnId sx <* notFollowedBy isDot)
              <|> try
                ( (,)
                    <$> (Just <$> (eitherP tableId alias' <* isDot))
                      <*> columnId sx
                )
          pure (distinct, tabAli, col)
    case at of
      Nothing -> p
      Just _ -> try (spaces *> betweenParentheses p) <|> try (someSpace *> p)
  v <-
    case tabAli of
      Just (Left t) -> do
        ParserEnv _ sqlSchema <- ask
        columnInTable <- guardTableColumn sx sqlSchema t col
        case columnInTable of
          TableNotInScope -> pure $ GuardTableColumn t col
          ColumnNotInTable ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show col
                       STC -> show col
                   )
                <> " is not in table "
                <> show t
          ColumnInTable -> pure $ GuardTableColumn t col
      Just (Right a) -> do
        ParserEnv _ sqlSchema <- ask
        columnInAlias <- guardAliasColumn sx sqlSchema a col
        case columnInAlias of
          AliasNotInScope -> pure $ GuardAliasColumn a col
          ColumnNotInAlias ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show col
                       STC -> show col
                   )
                <> " is not in alias "
                <> show a
          ColumnInAlias -> pure $ GuardAliasColumn a col
      Nothing -> pure $ GuardColumn col
  curScope <- (^. field @"psCurScope") <$> get
  case sx of
    SUD -> do
      modify (field @"psGuards" %~ HashMap.insertWith HashSet.union curScope (HashSet.singleton v))
      case distinct of
        Just _ -> pure $ DistinctColUnitUD at tabAli col
        Nothing -> pure $ ColUnitUD at tabAli col
    STC -> do
      modify (field @"psGuards" %~ HashMap.insertWith HashSet.union curScope (HashSet.singleton v))
      let colTyp Star = TStar
          colTyp (ColumnId typ _) = typ
      colUnitTyp' <- case at of
        Nothing -> pure $ colTyp col
        Just Count -> pure TNumber
        Just at'
          | colTyp col == TNumber -> pure TNumber
          | colTyp col == TTime -> pure TTime
          | colTyp col == TStar -> pure TStar
          | otherwise -> unexpected $ "unexpected column type " <> show (colTyp col) <> " for aggregation " <> show at'
      case distinct of
        Just _ -> pure $ DistinctColUnit colUnitTyp' at tabAli col
        Nothing -> pure $ ColUnit colUnitTyp' at tabAli col

-- | @inTable sqlSchema colId tabId@ checks if the 'ColumnId' @colId@ is valid for the table with the 'TableId' @tabId@ in the SQLSchema @sqlSchema@.
--
-- >>> flip testParse mempty $ inTable SUD sqlSchema (ColumnId () "Singer_ID") (TableId "song")
-- Done "" True
--
-- >>> flip testParse mempty $ inTable SUD sqlSchema (ColumnId () "singer_id") (TableId "song")
-- Done "" True
--
-- >>> flip testParse mempty $ inTable SUD sqlSchema (ColumnId () "Citizenship") (TableId "song")
-- Done "" False
inTable :: forall x m. (Parsing m, Monad m) => SX x -> Picard.Types.SQLSchema -> ColumnId x -> TableId -> m Bool
inTable _ _ Star _ = pure True
inTable sx Picard.Types.SQLSchema {..} ColumnId {..} TableId {..} =
  let matchingColumnUIds :: SX x -> m [Picard.Types.ColumnId]
      matchingColumnUIds SUD =
        pure
          . HashMap.keys
          . HashMap.filter (\x -> Text.toLower (Text.pack columnName) == Text.toLower x)
          $ sQLSchema_columnNames
      matchingColumnUIds STC = do
        columnTyp <- toColumnType columnIdX
        pure
          . HashMap.keys
          . HashMap.filter
            ( \(columnTyp', columnName') ->
                (Text.toLower (Text.pack columnName) == Text.toLower columnName')
                  && (columnTyp' == columnTyp)
            )
          $ HashMap.intersectionWith (,) sQLSchema_columnTypes sQLSchema_columnNames
      columnUIdToTableName =
        sQLSchema_tableNames
          `HashMap.compose` sQLSchema_columnToTable
      matchingTableNames = do
        columnUIds <- matchingColumnUIds sx
        pure . catMaybes $
          (`HashMap.lookup` columnUIdToTableName)
            <$> columnUIds
   in (Text.pack tableName `elem`) <$> matchingTableNames

-- | @inSelect SUD colId sel@ checks if the 'ColumnId' @colId@ is part of the 'Select' clause @sel@.
inSelect :: forall x. SX x -> ColumnId x -> Select x -> Bool
inSelect _ Star _ = True
inSelect sx c s =
  case s of
    Select _ aggs -> elemC sx aggs
    SelectDistinct _ aggs -> elemC sx aggs
  where
    elemC SUD aggs = c `elem` go aggs
    elemC STC aggs = c `elem` go aggs
    go [] = []
    go (Agg _ _ (Column _ (ValColUnit _ ColUnit {..})) : aggs) = colUnitColId : go aggs
    go (Agg _ _ (Column _ (ValColUnit _ DistinctColUnit {..})) : aggs) = distinctColUnitColdId : go aggs
    go (Agg {} : aggs) = go aggs

data GuardTableColumnResult
  = TableNotInScope
  | ColumnNotInTable
  | ColumnInTable
  deriving stock (Eq, Show)

guardTableColumn ::
  forall x m.
  ( Parsing m,
    MonadState (ParserState x) m
  ) =>
  SX x ->
  Picard.Types.SQLSchema ->
  TableId ->
  ColumnId x ->
  m GuardTableColumnResult
guardTableColumn sx sqlSchema t c = do
  ParserState {..} <- get
  if memberT psTables
    then do
      cInT <- c `inTable'` t
      if cInT
        then pure ColumnInTable
        else pure ColumnNotInTable
    else pure TableNotInScope
  where
    memberT = case sx of
      SUD -> (Left t `HashMap.member`)
      STC -> (Left t `HashMap.member`)
    inTable' = inTable sx sqlSchema

data GuardAliasColumnResult
  = AliasNotInScope
  | ColumnNotInAlias
  | ColumnInAlias
  deriving stock (Eq, Show)

guardAliasColumn ::
  forall x m.
  ( Parsing m,
    MonadState (ParserState x) m
  ) =>
  SX x ->
  Picard.Types.SQLSchema ->
  Alias ->
  ColumnId x ->
  m GuardAliasColumnResult
guardAliasColumn sx sqlSchema a c = do
  ParserState {..} <- get
  case HashMap.lookup a psAliases of
    Nothing -> pure AliasNotInScope
    Just m -> case Map.lookupLE psCurScope m of
      Nothing -> pure AliasNotInScope
      Just (_, TableUnitSQL SpiderSQL {..} _) ->
        pure $
          if inSelect sx c spiderSQLSelect
            then ColumnInAlias
            else ColumnNotInAlias
      Just (_, Table t _) ->
        let inTable' = inTable sx sqlSchema
         in do
              cInT <- c `inTable'` t
              if cInT
                then pure ColumnInAlias
                else pure ColumnNotInAlias

guardColumn ::
  forall x m.
  ( Parsing m,
    Monad m,
    MonadState (ParserState x) m
  ) =>
  SX x ->
  Picard.Types.SQLSchema ->
  ColumnId x ->
  m ()
guardColumn sx sqlSchema = go
  where
    inTable' = inTable sx sqlSchema
    go Star = pure ()
    go c = do
      ParserState {..} <- get
      columnInTablesPerScope <-
        Map.fromListWith (+) . join
          <$> traverse
            ( \case
                (Left t, scopes) -> do
                  columnInTable <- c `inTable'` t
                  pure $ (,fromEnum columnInTable) <$> HashSet.toList scopes
                (Right s, scopes) ->
                  pure $ (,fromEnum $ inSelect sx c s) <$> HashSet.toList scopes
            )
            (HashMap.toList psTables)
      columnInAliasesPerScope <-
        Map.fromListWith (+) . join
          <$> traverse
            ( \(_a, scopes) ->
                traverse
                  ( \(scope, tu) ->
                      case tu of
                        TableUnitSQL SpiderSQL {..} _ ->
                          pure (scope, fromEnum $ inSelect sx c spiderSQLSelect)
                        Table t _ -> do
                          columnInTable <- c `inTable'` t
                          pure (scope, fromEnum columnInTable)
                  )
                  (Map.toList scopes)
            )
            (HashMap.toList psAliases)
      let columnPerScope = Map.unionWith (+) columnInTablesPerScope columnInAliasesPerScope
      unless
        ( maybe False ((== 1) . snd) $
            Map.lookupLE psCurScope columnPerScope
        )
        . unexpected
        $ "there is no single table in scope with column "
          <> ( case sx of
                 SUD -> show c
                 STC -> show c
             )
          <> "."

-- | @withGuards sqlSchema p@ fails conditioned on whether or not
-- all referenced columns are members of tables or aliases that are in scope.
withGuards ::
  forall x m p.
  ( Parsing m,
    MonadPlus m,
    MonadState (ParserState x) m
  ) =>
  SX x ->
  Picard.Types.SQLSchema ->
  m p ->
  m p
withGuards sx sqlSchema p = do
  pRes <- p
  ParserState {..} <- get
  let curGuards =
        fromMaybe
          ( case sx of
              SUD -> mempty
              STC -> mempty
          )
          $ HashMap.lookup psCurScope psGuards
      f :: Guard x -> m ()
      f (GuardTableColumn t c) = do
        columnInTable <- guardTableColumn sx sqlSchema t c
        case columnInTable of
          TableNotInScope ->
            unexpected $
              "table "
                <> show t
                <> " is not in scope."
          ColumnNotInTable ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show c
                       STC -> show c
                   )
                <> " is not in table "
                <> show t
                <> "."
          ColumnInTable -> pure ()
      f (GuardAliasColumn a c) = do
        columnInAlias <- guardAliasColumn sx sqlSchema a c
        case columnInAlias of
          AliasNotInScope ->
            unexpected $
              "alias "
                <> show a
                <> " is not in scope."
          ColumnNotInAlias ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show c
                       STC -> show c
                   )
                <> " is not in alias "
                <> show a
                <> "."
          ColumnInAlias -> pure ()
      f (GuardColumn c) = guardColumn sx sqlSchema c
  forM_ curGuards f
  pure pRes

-- | 'TableId' parser.
--
-- >>> testParseOnly tableId "singer"
-- Right (TableId {tableName = "singer"})
--
-- >>> testParseOnly tableId "Singer"
-- Right (TableId {tableName = "singer"})
--
-- >>> testParseOnly tableId "sanger"
-- Left "tableId: Failed reading: mzero"
--
-- >>> (Atto.parseOnly . flip runReaderT (ParserEnv (ParserEnvWithGuards (withGuards SUD)) sqlSchema { sQLSchema_tableNames = HashMap.union (sQLSchema_tableNames sqlSchema) (HashMap.singleton "2" "singers") }) . flip evalStateT mkParserStateUD) (tableId <* (lift . lift) Atto.endOfInput) "singers"
-- Right (TableId {tableName = "singers"})
tableId :: forall x m. (CharParsing m, MonadReader (ParserEnv x) m, MonadPlus m) => m TableId
tableId =
  let terminate q = try q <* notFollowedBy (try alphaNum <|> try (char '_'))
   in flip (<?>) "tableId" $ do
        ParserEnv _ sqlSchema <- ask
        let tableNameP' = runReaderT tableNameP sqlSchema
        TableId <$> terminate tableNameP'

-- | 'Alias' parser.
--
-- Hardcoded to start with an alphabetic character and to be at most 10 characters long.
alias :: forall x m. (CharParsing m, MonadReader (ParserEnv x) m, MonadPlus m) => m Alias
alias =
  let terminate q = q <* notFollowedBy (try alphaNum <|> try (char '_'))
      name =
        let p = try $ satisfy ((||) <$> isAlphaNum <*> (== '_'))
         in liftA2 (:) (satisfy isAlpha) (manyAtMost (9 :: Int) p)
   in flip (<?>) "alias" $ do
        ParserEnv _ sqlSchema <- ask
        let tableNameP' = runReaderT tableNameP sqlSchema
            columnNameP' = runReaderT columnNameP sqlSchema
        Alias
          <$> ( try (columnNameP' *> unexpected "alias must not be column name")
                  <|> try (tableNameP' *> unexpected "alias must not be table name")
                  <|> try (terminate name)
              )

-- | Alternative 'Alias' parser.
--
-- Hardcoded to start with a T followed by at most 9 digits.
alias' :: (CharParsing m, Monad m) => m Alias
alias' = flip (<?>) "alias" $ do
  _ <- try $ satisfy (\c -> toLower c == 't')
  let terminate q = try q <* notFollowedBy (try alphaNum <|> try (char '_'))
  digits <- terminate $ liftA2 (:) digit (manyAtMost (9 :: Int) digit)
  pure . Alias $ "T" <> digits

-- | 'ColumnId' parser.
--
-- >>> testParseOnly (columnId SUD) "*"
-- Right Star
--
-- >>> testParseOnly (columnId SUD) "invalid_column"
-- Left "columnId: Failed reading: mzero"
--
-- >>> testParseOnly (columnId SUD) "Birth_Year"
-- Right (ColumnId {columnIdX = (), columnName = "Birth_Year"})
--
-- >>> testParseOnly (columnId SUD) "birth_year"
-- Right (ColumnId {columnIdX = (), columnName = "Birth_Year"})
columnId :: forall x m. (CharParsing m, MonadReader (ParserEnv x) m, MonadPlus m) => SX x -> m (ColumnId x)
columnId sx =
  let terminate q = q <* notFollowedBy (try alphaNum <|> try (char '_'))
   in flip (<?>) "columnId" $ do
        ParserEnv _ sqlSchema <- ask
        case sx of
          SUD ->
            let columnNameP' = runReaderT columnNameP sqlSchema
             in try (isStar $> Star) <|> try (ColumnIdUD <$> terminate columnNameP')
          STC ->
            let columnTypeAndNameP' = runReaderT columnTypeAndNameP sqlSchema
             in try (isStar $> Star) <|> try (uncurry ColumnId <$> terminate columnTypeAndNameP')

-- | 'From' parser.
--
-- >>> testParseOnly (from SUD) "FROM singer"
-- Right (From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing})
--
-- >>> testParseOnly (from SUD) "FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID"
-- Right (From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))})
from ::
  forall x m.
  ( TokenParsing m,
    MonadSQL x m
  ) =>
  SX x ->
  m (From x)
from sx = flip (<?>) "from" $ do
  _ <- isFrom
  someSpace
  uncurry mkFrom <$> p
  where
    p :: m (TableUnit x, [(TableUnit x, Maybe (Cond x))])
    p =
      (,)
        <$> tableUnit sx
        <*> many
          ( try $
              someSpace
                *> isJoin
                *> someSpace
                *> ( (,)
                       <$> tableUnit sx
                       <*> optional
                         ( try $
                             someSpace
                               *> isOn
                               *> someSpace
                               *> cond sx
                         )
                   )
          )
    mkFrom :: TableUnit x -> [(TableUnit x, Maybe (Cond x))] -> From x
    mkFrom tu tus =
      From
        (tu : fmap fst tus)
        ( foldl'
            ( \a b ->
                case (a, b) of
                  (Just c, Just c') -> Just (And c c')
                  (Just c, Nothing) -> Just c
                  (Nothing, Just c') -> Just c'
                  (Nothing, Nothing) -> Nothing
            )
            Nothing
            (fmap snd tus)
        )

updateAliases ::
  forall x m.
  ( Parsing m,
    MonadSQL x m
  ) =>
  SX x ->
  Alias ->
  TableUnit x ->
  m ()
updateAliases sx a tu = do
  ParserState {..} <- get
  hasConflict <-
    maybe False (Map.member psCurScope)
      . HashMap.lookup a
      . (^. field @"psAliases")
      <$> get
  when hasConflict
    . unexpected
    $ "the alias "
      <> show a
      <> "is already in this scope."
  let v = Map.singleton psCurScope tu
  modify (field @"psAliases" %~ HashMap.insertWith Map.union a v)
  let curGuards =
        fromMaybe
          ( case sx of
              SUD -> mempty
              STC -> mempty
          )
          $ HashMap.lookup psCurScope psGuards
      f (GuardAliasColumn a' c) | a == a' = do
        ParserEnv _ sqlSchema <- ask
        columnInAlias <- guardAliasColumn sx sqlSchema a c
        case columnInAlias of
          AliasNotInScope -> error "impossible"
          ColumnNotInAlias ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show c
                       STC -> show c
                   )
                <> " is not in alias "
                <> show a
                <> "."
          ColumnInAlias -> pure ()
      f _ = pure ()
  forM_ curGuards f

updateTables ::
  forall x m.
  ( Parsing m,
    MonadSQL x m
  ) =>
  SX x ->
  Either TableId (Select x) ->
  m ()
updateTables sx (Left t) = do
  ParserState {..} <- get
  hasConflict <-
    maybe False (HashSet.member psCurScope)
      . ( case sx of
            SUD -> HashMap.lookup (Left t)
            STC -> HashMap.lookup (Left t)
        )
      . (^. field @"psTables")
      <$> get
  when hasConflict
    . unexpected
    $ "the table "
      <> show t
      <> "is already in this scope."
  let v = HashSet.singleton psCurScope
  modify
    ( field @"psTables"
        %~ ( case sx of
               SUD -> HashMap.insertWith HashSet.union (Left t) v
               STC -> HashMap.insertWith HashSet.union (Left t) v
           )
    )
  let curGuards =
        fromMaybe
          ( case sx of
              SUD -> mempty
              STC -> mempty
          )
          $ HashMap.lookup psCurScope psGuards
      f (GuardTableColumn t' c) | t == t' = do
        ParserEnv _ sqlSchema <- ask
        columnInTable <- guardTableColumn sx sqlSchema t c
        case columnInTable of
          TableNotInScope -> error "impossible"
          ColumnNotInTable ->
            unexpected $
              "column "
                <> ( case sx of
                       SUD -> show c
                       STC -> show c
                   )
                <> " is not in table "
                <> show t
                <> "."
          ColumnInTable -> pure ()
      f _ = pure ()
  forM_ curGuards f
updateTables sx (Right s) = do
  ParserState {..} <- get
  let v = HashSet.singleton psCurScope
  modify
    ( field @"psTables"
        %~ ( case sx of
               SUD -> HashMap.insertWith HashSet.union (Right s) v
               STC -> HashMap.insertWith HashSet.union (Right s) v
           )
    )

-- | 'TableUnit' parser.
--
-- >>> testParseOnly (tableUnit SUD) "song as t1"
-- Right (Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T1"})))
--
-- >>> testParseOnly (tableUnit SUD) "(SELECT * FROM song)"
-- Right (TableUnitSQL (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}) Nothing)
--
-- >>> testParseOnly (tableUnit SUD) "(SELECT * FROM song) as t1"
-- Right (TableUnitSQL (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}) (Just (Alias {aliasName = "T1"})))
tableUnit ::
  forall x m.
  ( TokenParsing m,
    MonadSQL x m
  ) =>
  SX x ->
  m (TableUnit x)
tableUnit sx =
  flip (<?>) "tableUnit" $
    let aliasP = someSpace *> isAs *> someSpace *> alias'
        tableUnitSQL =
          flip (<?>) "tableUnitSQL" $
            TableUnitSQL
              <$> betweenParentheses (get >>= spiderSQL sx . (field @"psCurScope" %~ succ))
                <*> optional (try aliasP)
        table =
          flip (<?>) "table" $
            Table
              <$> tableId
                <*> optional (try aliasP)
     in do
          tu <- try tableUnitSQL <|> try table
          case tu of
            TableUnitSQL _ (Just a) -> updateAliases sx a tu
            TableUnitSQL SpiderSQL {..} Nothing -> updateTables sx (Right spiderSQLSelect)
            Table _ (Just a) -> updateAliases sx a tu
            Table t Nothing -> updateTables sx (Left t)
          pure tu

-- | 'Cond' parser.
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID = t2.Singer_ID"
-- Right (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID + t2.Singer_ID = t2.Singer_ID"
-- Right (Eq (Plus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID = t2.Singer_ID + t2.Singer_ID"
-- Right (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Plus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly (cond SUD) "t2.Name = \"Adele\" AND t3.Name = \"Beyoncé\""
-- Right (And (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}})) (Column () (ValString {stringValue = "Adele"}))) (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T3"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}})) (Column () (ValString {stringValue = "Beyonc\233"}))))
--
-- >>> testParseOnly (cond SUD) "song_id IN (SELECT song_id FROM song)"
-- Right (In (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly (cond SUD) "song_id NOT IN (SELECT song_id FROM song)"
-- Right (Not (In (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}}))))
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID - t2.Singer_ID = (select song_id - song_id from song order by song_id - song_id desc)"
-- Right (Eq (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID - t2.Singer_ID = (select song_id - song_id from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly (cond SUD) "t1.Singer_ID - t2.Singer_ID = (select (song_id - song_id) from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly (cond SUD) "(t1.Singer_ID - t2.Singer_ID) = (select (song_id - song_id) from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}) (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
cond :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (Cond x)
cond sx =
  flip (<?>) "cond" $
    let mkCond p' =
          let suffix r' =
                let q = mkCond p'
                 in choice
                      [ try $ And r' <$> (someSpace *> isAnd *> someSpace *> q),
                        try $ Or r' <$> (someSpace *> isOr *> someSpace *> q)
                      ]
              suffixRec base = do
                c <- base
                try (suffixRec (suffix c)) <|> try (pure c)
              r =
                choice
                  [ try $ Not <$> (isNot *> spaces *> p'),
                    try p'
                  ]
           in suffixRec r
        starEq TStar _ = True
        starEq _ TStar = True
        starEq typ typ' = typ == typ'
        checkedEquality :: SX x -> (ValUnit x -> ValUnit x -> Cond x) -> ValUnit x -> ValUnit x -> m (Cond x)
        checkedEquality SUD f vu vu' = pure $ f vu vu'
        checkedEquality STC f vu vu'
          | valUnitTyp vu `starEq` valUnitTyp vu' = pure $ f vu vu'
          | otherwise = unexpected $ "the types of " <> show vu <> " and " <> show vu' <> " do not work for equality conditions"
        checkedNumerical :: SX x -> (ValUnit x -> ValUnit x -> Cond x) -> ValUnit x -> ValUnit x -> m (Cond x)
        checkedNumerical SUD f vu vu' = pure $ f vu vu'
        checkedNumerical STC f vu vu'
          | valUnitTyp vu `starEq` TNumber && valUnitTyp vu' `starEq` TNumber = pure $ f vu vu'
          | valUnitTyp vu `starEq` TTime && valUnitTyp vu' `starEq` TTime = pure $ f vu vu'
          | otherwise = unexpected $ "the types of " <> show vu <> " and " <> show vu' <> " do not work for numerical comparison conditions"
        checkedLikeness :: SX x -> (ValUnit x -> ValUnit x -> Cond x) -> ValUnit x -> ValUnit x -> m (Cond x)
        checkedLikeness SUD f vu vu' = pure $ f vu vu'
        checkedLikeness STC f vu vu'
          | valUnitTyp vu `starEq` TText && valUnitTyp vu' `starEq` TText = pure $ f vu vu'
          | otherwise = unexpected $ "the types of " <> show vu <> " and " <> show vu' <> " do not work for like conditions"
        checkedBetween :: SX x -> ValUnit x -> ValUnit x -> ValUnit x -> m (Cond x)
        checkedBetween SUD vu vu' vu'' = pure $ Between vu vu' vu''
        checkedBetween STC vu vu' vu''
          | valUnitTyp vu `starEq` TNumber && valUnitTyp vu' `starEq` TNumber && valUnitTyp vu'' `starEq` TNumber = pure $ Between vu vu' vu''
          | valUnitTyp vu `starEq` TTime && valUnitTyp vu' `starEq` TTime && valUnitTyp vu'' `starEq` TTime = pure $ Between vu vu' vu''
          | otherwise = unexpected $ "the types of " <> show vu <> ", " <> show vu' <> ", and " <> show vu'' <> " do not work for between conditions"
        p =
          choice
            [ try $ binary (checkedEquality sx Eq) isEq,
              try $ binary (checkedNumerical sx Ge) isGe,
              try $ binary (checkedNumerical sx Le) isLe,
              try $ binary (checkedNumerical sx Gt) isGt,
              try $ binary (checkedNumerical sx Lt) isLt,
              try $ binary (checkedEquality sx Ne) isNe,
              try $ binary (checkedEquality sx In) isIn,
              try $ binaryNot (checkedEquality sx In) isIn,
              try $ binary (checkedLikeness sx Like) isLike,
              try $ binaryNot (checkedLikeness sx Like) isLike,
              try between'
            ]
        binary f q = do
          vu <- betweenOptionalParentheses (valUnit sx)
          vu' <- spaces *> q *> spaces *> betweenOptionalParentheses (valUnit sx)
          f vu vu'
        binaryNot f q = do
          vu <- betweenOptionalParentheses (valUnit sx)
          vu' <- spaces *> isNot *> someSpace *> q *> spaces *> betweenOptionalParentheses (valUnit sx)
          Not <$> f vu vu'
        between' = do
          vu <- betweenOptionalParentheses (valUnit sx)
          vu' <- someSpace *> isBetween *> someSpace *> betweenOptionalParentheses (valUnit sx)
          vu'' <- someSpace *> isAnd *> someSpace *> betweenOptionalParentheses (valUnit sx)
          checkedBetween sx vu vu' vu''
     in mkCond p

-- | 'Val' parser.
--
-- >>> testParseOnly (val SUD) "count song.Song_ID"
-- Right (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Left (TableId {tableName = "song"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})
--
-- >>> testParseOnly (val SUD) "count(song.Song_ID)"
-- Right (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Left (TableId {tableName = "song"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})
--
-- >>> testParseOnly (val SUD) "(select *)"
-- Right (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})
--
-- >>> testParseOnly (val SUD) "(select song_id from song)"
-- Right (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})
val ::
  forall x m.
  ( TokenParsing m,
    MonadSQL x m
  ) =>
  SX x ->
  m (Val x)
val sx = flip (<?>) "val" $ choice choices
  where
    terminate q = q <* notFollowedBy (try alphaNum <|> try (char '_'))
    choices = [try valColUnit, try number, try valString, try valSQL]
    valColUnit = do
      cu <- colUnit sx
      pure $ ValColUnit (colUnitTyp cu) cu
    number = Number <$> terminate (doubleP 16)
    valString = ValString <$> terminate (quotedString 64)
    valSQL = do
      sql <- betweenParentheses (get >>= spiderSQL sx . (field @"psCurScope" %~ succ))
      pure $ ValSQL (spiderSQLTyp sql) sql

-- | Parser for WHERE clauses.
--
-- >>> testParseOnly (whereCond SUD) "where t1.Singer_ID = t2.Singer_ID"
-- Right (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly (whereCond SUD) "where Singer_ID = 1"
-- Right (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (Number {numberValue = 1.0})))
whereCond :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (Cond x)
whereCond sx = flip (<?>) "whereCond" $ isWhere *> someSpace *> cond sx

-- | Parser for group-by clauses.
--
-- >>> testParseOnly (groupBy SUD) "group by t1.Song_ID"
-- Right [ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}]
--
-- >>> testParseOnly (groupBy SUD) "group by t1.Song_ID, t2.Singer_ID"
-- Right [ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}},ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}]
--
-- >>> testParseOnly (groupBy SUD) "group by count t1.Song_ID, t2.Singer_ID"
-- Right [ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}},ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}]
groupBy ::
  forall x m.
  ( TokenParsing m,
    MonadSQL x m
  ) =>
  SX x ->
  m [ColUnit x]
groupBy sx =
  flip (<?>) "groupBy" $
    isGroupBy
      *> someSpace
      *> sepBy1 (try $ betweenOptionalParentheses (colUnit sx)) (try $ spaces *> isComma <* someSpace)

-- | 'OrderBy' Parser.
--
-- >>> testParseOnly (orderBy SUD) "order by t1.Song_ID, t2.Singer_ID desc"
-- Right (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Asc),(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}),Desc)])
--
-- >>> testParseOnly (orderBy SUD) "order by t1.Song_ID asc, t2.Singer_ID desc"
-- Right (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Asc),(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}}),Desc)])
--
-- >>> testParseOnly (orderBy SUD) "order by count(t1.Song_ID) desc"
-- Right (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Desc)])
--
-- >>> testParseOnly (orderBy SUD) "order by sum(t1.Song_ID)"
-- Right (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Sum, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}),Asc)])
orderBy :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (OrderBy x)
orderBy sx = flip (<?>) "orderBy" $ do
  _ <- isOrderBy
  someSpace
  valUnits <-
    let order = optional (try $ spaces *> (try (isAsc $> Asc) <|> try (isDesc $> Desc))) >>= maybe (pure Asc) pure
        p = (,) <$> betweenOptionalParentheses (valUnit sx) <*> order
     in sepBy1 (try p) (try $ spaces *> isComma <* someSpace)
  pure $ OrderBy valUnits

-- | Parser for HAVING clauses.
--
-- >>> testParseOnly (havingCond SUD) "having count(t1.Sales) = 10"
-- Right (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Sales"}}})) (Column () (Number {numberValue = 10.0})))
havingCond :: forall x m. (TokenParsing m, MonadSQL x m) => SX x -> m (Cond x)
havingCond sx = flip (<?>) "havingCond" $ isHaving *> someSpace *> cond sx

-- | Parser for LIMIT clauses.
--
-- >>> testParseOnly limit "limit 10"
-- Right 10
limit :: forall m. (TokenParsing m, Monad m) => m Int
limit = flip (<?>) "limit" $ isLimit *> someSpace *> intP 8

-- | 'SpiderSQL' parser.
--
-- >>> spiderSQLTestParseOnly "select *"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select count(*)"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select * from singer"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select * from song"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T1.Name, T1.Citizenship, T1.Birth_Year FROM singer AS T1 ORDER BY T1.Birth_Year DESC"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Citizenship"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"}))], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT Name, Citizenship, Birth_Year FROM singer ORDER BY Birth_Year DESC"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Citizenship"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT   name  ,    citizenship  ,   birth_year   FROM   Singer  ORDER BY   birth_year   DESC"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Citizenship"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T2.Title, T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Title"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Name HAVING COUNT(*) > 1"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Name"}}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Just (Gt (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star}})) (Column () (Number {numberValue = 1.0}))), spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select count t2.Song_ID, t1.Citizenship from singer AS t1 JOIN song AS t2 on t1.Singer_ID = t2.Singer_ID group by count t2.Song_ID, t1.Citizenship"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})),Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Citizenship"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitX = (), colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}},ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Citizenship"}}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT title FROM song WHERE song_id IN (SELECT song_id FROM song)"
-- Right (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Title"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Just (In (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}})) (Column () (ValSQL {sqlValueX = (), sqlValue = SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnIdX = (), columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}}))), spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParse "SELECT T1.title ,  count(*) FROM song AS T1 JOIN song AS T2 ON T1.song id"
-- Done " ON T1.song id" (SpiderSQL {spiderSQLX = (), spiderSQLSelect = Select () [Agg () Nothing (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnIdX = (), columnName = "Title"}}})),Agg () (Just Count) (Column () (ValColUnit {columnX = (), columnValue = ColUnit {colUnitX = (), colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
spiderSQL ::
  forall x m.
  ( TokenParsing m,
    MonadPlus m,
    MonadReader (ParserEnv x) m
  ) =>
  SX x ->
  ParserState x ->
  m (SpiderSQL x)
spiderSQL sx env =
  flip (<?>) "spiderSQL" $
    flip evalStateT env $ do
      ParserEnv (ParserEnvWithGuards peWithGuards) sqlSchema <- ask
      sel <- select sx
      fro <- peWithGuards sqlSchema $ fromMaybe (From [] Nothing) <$> optional (try $ spaces *> from sx)
      whe <- optional (try . peWithGuards sqlSchema $ someSpace *> whereCond sx)
      grp <- fromMaybe [] <$> optional (try . peWithGuards sqlSchema $ someSpace *> groupBy sx)
      (ord, hav) <-
        permute
          ( (,) <$$> try (optional (try . peWithGuards sqlSchema $ someSpace *> orderBy sx))
              <||> try (optional (try . peWithGuards sqlSchema $ someSpace *> havingCond sx))
          )
      lim <- optional (try $ someSpace *> limit)
      (int, exc, uni) <-
        permute
          ( (,,) <$$> try (optional (try $ someSpace *> isIntersect *> someSpace *> spiderSQL sx env))
              <||> try (optional (try $ someSpace *> isExcept *> someSpace *> spiderSQL sx env))
              <||> try (optional (try $ someSpace *> isUnion *> someSpace *> spiderSQL sx env))
          )
      case sx of
        SUD -> pure $ SpiderSQLUD sel fro whe grp ord hav lim int exc uni
        STC -> do
          let typGuard x
                | selectTyp sel == spiderSQLTyp x = pure ()
                | otherwise = unexpected $ show sel <> " and " <> show x <> " have incompatible types"
          for_ int typGuard
          for_ exc typGuard
          for_ uni typGuard
          pure $ SpiderSQL (selectTyp sel) sel fro whe grp ord hav lim int exc uni
