{-# LANGUAGE EmptyDataDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeFamilies #-}

module Language.SQL.SpiderSQL.Syntax where

import Data.Hashable (Hashable)
import Data.Kind (Type)
import GHC.Generics (Generic)

data SpiderTyp
  = TBoolean
  | TNumber
  | TText
  | TOthers
  | TTime
  | TStar
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data X = UD | TC

data SX :: X -> Type where
  SUD :: SX 'UD
  STC :: SX 'TC

type XSpiderSQL :: X -> Type
type family XSpiderSQL x where
  XSpiderSQL 'UD = ()
  XSpiderSQL 'TC = [SpiderTyp]

type SpiderSQL :: X -> Type
data SpiderSQL x = SpiderSQL
  { spiderSQLX :: XSpiderSQL x,
    spiderSQLSelect :: Select x,
    spiderSQLFrom :: From x,
    spiderSQLWhere :: Maybe (Cond x),
    spiderSQLGroupBy :: [ColUnit x],
    spiderSQLOrderBy :: Maybe (OrderBy x),
    spiderSQLHaving :: Maybe (Cond x),
    spiderSQLLimit :: Maybe Int,
    spiderSQLIntersect :: Maybe (SpiderSQL x),
    spiderSQLExcept :: Maybe (SpiderSQL x),
    spiderSQLUnion :: Maybe (SpiderSQL x)
  }
  deriving stock (Generic)

type SpiderSQLUD = SpiderSQL 'UD

pattern SpiderSQLUD ::
  SelectUD ->
  FromUD ->
  Maybe CondUD ->
  [ColUnitUD] ->
  Maybe OrderByUD ->
  Maybe CondUD ->
  Maybe Int ->
  Maybe SpiderSQLUD ->
  Maybe SpiderSQLUD ->
  Maybe SpiderSQLUD ->
  SpiderSQLUD
pattern SpiderSQLUD sqlSelect sqlFrom sqlWhere sqlGroupBy sqlOrderBy sqlHaving sqlLimit sqlIntersect sqlExcept sqlUnion <-
  SpiderSQL _ sqlSelect sqlFrom sqlWhere sqlGroupBy sqlOrderBy sqlHaving sqlLimit sqlIntersect sqlExcept sqlUnion
  where
    SpiderSQLUD sqlSelect sqlFrom sqlWhere sqlGroupBy sqlOrderBy sqlHaving sqlLimit sqlIntersect sqlExcept sqlUnion =
      SpiderSQL () sqlSelect sqlFrom sqlWhere sqlGroupBy sqlOrderBy sqlHaving sqlLimit sqlIntersect sqlExcept sqlUnion

deriving stock instance Eq SpiderSQLUD

deriving stock instance Show SpiderSQLUD

deriving anyclass instance Hashable SpiderSQLUD

spiderSQLTyp :: SpiderSQL x -> XSpiderSQL x
spiderSQLTyp SpiderSQL {..} = spiderSQLX

type SpiderSQLTC = SpiderSQL 'TC

deriving stock instance Eq SpiderSQLTC

deriving stock instance Show SpiderSQLTC

deriving anyclass instance Hashable SpiderSQLTC

type family XSelect x where
  XSelect 'UD = ()
  XSelect 'TC = [SpiderTyp]

data Select x
  = Select (XSelect x) [Agg x]
  | SelectDistinct (XSelect x) [Agg x]
  deriving stock (Generic)

pattern SelectUD :: [AggUD] -> SelectUD
pattern SelectUD aggs <-
  Select _ aggs
  where
    SelectUD aggs = Select () aggs

pattern SelectDistinctUD :: [AggUD] -> SelectUD
pattern SelectDistinctUD aggs <-
  SelectDistinct _ aggs
  where
    SelectDistinctUD aggs = SelectDistinct () aggs

type SelectUD = Select 'UD

deriving stock instance Eq SelectUD

deriving stock instance Show SelectUD

deriving anyclass instance Hashable SelectUD

selectTyp :: Select x -> XSelect x
selectTyp (Select typ _) = typ
selectTyp (SelectDistinct typ _) = typ

type SelectTC = Select 'TC

deriving stock instance Eq SelectTC

deriving stock instance Show SelectTC

deriving anyclass instance Hashable SelectTC

data From x = From
  { fromTableUnits :: [TableUnit x],
    fromCond :: Maybe (Cond x)
  }
  deriving stock (Generic)

type FromUD = From 'UD

deriving stock instance Eq FromUD

deriving stock instance Show FromUD

deriving anyclass instance Hashable FromUD

type FromTC = From 'TC

deriving stock instance Eq FromTC

deriving stock instance Show FromTC

deriving anyclass instance Hashable FromTC

type Cond :: X -> Type
data Cond x
  = And (Cond x) (Cond x)
  | Or (Cond x) (Cond x)
  | Not (Cond x)
  | Between (ValUnit x) (ValUnit x) (ValUnit x)
  | Eq (ValUnit x) (ValUnit x)
  | Gt (ValUnit x) (ValUnit x)
  | Lt (ValUnit x) (ValUnit x)
  | Ge (ValUnit x) (ValUnit x)
  | Le (ValUnit x) (ValUnit x)
  | Ne (ValUnit x) (ValUnit x)
  | In (ValUnit x) (ValUnit x)
  | Like (ValUnit x) (ValUnit x)
  deriving stock (Generic)

type CondUD = Cond 'UD

deriving stock instance Eq CondUD

deriving stock instance Show CondUD

deriving anyclass instance Hashable CondUD

type CondTC = Cond 'TC

deriving stock instance Eq CondTC

deriving stock instance Show CondTC

deriving anyclass instance Hashable CondTC

type family XValUnit x where
  XValUnit 'UD = ()
  XValUnit 'TC = SpiderTyp

type ValUnit :: X -> Type
data ValUnit x
  = Column (XValUnit x) (Val x)
  | Minus (XValUnit x) (Val x) (Val x)
  | Plus (XValUnit x) (Val x) (Val x)
  | Times (XValUnit x) (Val x) (Val x)
  | Divide (XValUnit x) (Val x) (Val x)
  deriving stock (Generic)

valUnitTyp :: forall x. ValUnit x -> XValUnit x
valUnitTyp (Column typ _) = typ
valUnitTyp (Minus typ _ _) = typ
valUnitTyp (Plus typ _ _) = typ
valUnitTyp (Times typ _ _) = typ
valUnitTyp (Divide typ _ _) = typ

pattern ColumnUD :: ValUD -> ValUnitUD
pattern ColumnUD val <-
  Column _ val
  where
    ColumnUD val = Column () val

type ValUnitUD = ValUnit 'UD

deriving stock instance Eq ValUnitUD

deriving stock instance Show ValUnitUD

deriving anyclass instance Hashable ValUnitUD

type ValUnitTC = ValUnit 'TC

deriving stock instance Eq ValUnitTC

deriving stock instance Show ValUnitTC

deriving anyclass instance Hashable ValUnitTC

type XValColUnit :: X -> Type

type XValColUnit x = XColUnit x

type XValSQL :: X -> Type

type XValSQL x = XSpiderSQL x

type Val :: X -> Type
data Val x
  = ValColUnit {columnX :: XValColUnit x, columnValue :: ColUnit x}
  | Number {numberValue :: Double}
  | ValString {stringValue :: String}
  | ValSQL {sqlValueX :: XValSQL x, sqlValue :: SpiderSQL x}
  deriving stock (Generic)

type ValUD = Val 'UD

deriving stock instance Eq ValUD

deriving stock instance Show ValUD

deriving anyclass instance Hashable ValUD

type ValTC = Val 'TC

deriving stock instance Eq ValTC

deriving stock instance Show ValTC

deriving anyclass instance Hashable ValTC

type XColUnit :: X -> Type
type family XColUnit x where
  XColUnit 'UD = ()
  XColUnit 'TC = SpiderTyp

data ColUnit x
  = ColUnit
      { colUnitX :: XColUnit x,
        colUnitAggId :: Maybe AggType,
        colUnitTable :: Maybe (Either TableId Alias),
        colUnitColId :: ColumnId x
      }
  | DistinctColUnit
      { distinctColUnitX :: XColUnit x,
        distinctColUnitAggId :: Maybe AggType,
        distinctColUnitTable :: Maybe (Either TableId Alias),
        distinctColUnitColdId :: ColumnId x
      }
  deriving stock (Generic)

pattern ColUnitUD :: Maybe AggType -> Maybe (Either TableId Alias) -> ColumnIdUD -> ColUnitUD
pattern ColUnitUD at tidOrAlias cid <-
  ColUnit _ at tidOrAlias cid
  where
    ColUnitUD at tidOrAlias cid = ColUnit () at tidOrAlias cid

pattern DistinctColUnitUD :: Maybe AggType -> Maybe (Either TableId Alias) -> ColumnIdUD -> ColUnitUD
pattern DistinctColUnitUD at tidOrAlias cid <-
  DistinctColUnit _ at tidOrAlias cid
  where
    DistinctColUnitUD at tidOrAlias cid = DistinctColUnit () at tidOrAlias cid

type ColUnitUD = ColUnit 'UD

deriving stock instance Eq ColUnitUD

deriving stock instance Show ColUnitUD

deriving anyclass instance Hashable ColUnitUD

colUnitTyp :: forall x. ColUnit x -> XColUnit x
colUnitTyp ColUnit {..} = colUnitX
colUnitTyp DistinctColUnit {..} = distinctColUnitX

type ColUnitTC = ColUnit 'TC

deriving stock instance Eq ColUnitTC

deriving stock instance Show ColUnitTC

deriving anyclass instance Hashable ColUnitTC

type OrderBy :: X -> Type
newtype OrderBy x = OrderBy [(ValUnit x, OrderByOrder)]
  deriving stock (Generic)

type OrderByUD = OrderBy 'UD

deriving stock instance Eq OrderByUD

deriving stock instance Show OrderByUD

deriving anyclass instance Hashable OrderByUD

type OrderByTC = OrderBy 'TC

deriving stock instance Eq OrderByTC

deriving stock instance Show OrderByTC

deriving anyclass instance Hashable OrderByTC

data OrderByOrder = Asc | Desc
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

type XAgg :: X -> Type
type family XAgg x where
  XAgg 'UD = ()
  XAgg 'TC = SpiderTyp

type Agg :: X -> Type
data Agg x = Agg (XAgg x) (Maybe AggType) (ValUnit x)
  deriving stock (Generic)

pattern AggUD :: Maybe AggType -> ValUnitUD -> AggUD
pattern AggUD aggType vu <-
  Agg _ aggType vu
  where
    AggUD aggType vu = Agg () aggType vu

type AggUD = Agg 'UD

deriving stock instance Eq AggUD

deriving stock instance Show AggUD

deriving anyclass instance Hashable AggUD

type AggTC = Agg 'TC

deriving stock instance Eq AggTC

deriving stock instance Show AggTC

deriving anyclass instance Hashable AggTC

type TableUnit :: X -> Type
data TableUnit x
  = TableUnitSQL (SpiderSQL x) (Maybe Alias)
  | Table TableId (Maybe Alias)
  deriving stock (Generic)

type TableUnitUD = TableUnit 'UD

deriving stock instance Eq TableUnitUD

deriving stock instance Show TableUnitUD

deriving anyclass instance Hashable TableUnitUD

type TableUnitTC = TableUnit 'TC

deriving stock instance Eq TableUnitTC

deriving stock instance Show TableUnitTC

deriving anyclass instance Hashable TableUnitTC

data AggType = Max | Min | Count | Sum | Avg
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

type XColumnID :: X -> Type
type family XColumnID x where
  XColumnID 'UD = ()
  XColumnID 'TC = SpiderTyp

data ColumnId x
  = Star
  | ColumnId {columnIdX :: XColumnID x, columnName :: String}
  deriving stock (Generic)

pattern ColumnIdUD :: String -> ColumnIdUD
pattern ColumnIdUD colName <-
  ColumnId _ colName
  where
    ColumnIdUD colName = ColumnId () colName

type ColumnIdUD = ColumnId 'UD

deriving stock instance Eq ColumnIdUD

deriving stock instance Show ColumnIdUD

deriving anyclass instance Hashable ColumnIdUD

type ColumnIdTC = ColumnId 'TC

deriving stock instance Eq ColumnIdTC

deriving stock instance Show ColumnIdTC

deriving anyclass instance Hashable ColumnIdTC

newtype TableId = TableId {tableName :: String}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)

newtype Alias = Alias {aliasName :: String}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)
