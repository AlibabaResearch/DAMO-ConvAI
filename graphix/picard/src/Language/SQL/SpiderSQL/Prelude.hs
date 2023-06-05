module Language.SQL.SpiderSQL.Prelude where

import Control.Applicative (Alternative (empty, (<|>)), Applicative (liftA2))
import Control.Monad.Reader.Class (MonadReader (ask))
import Data.Char (toLower)
import Data.Function (on)
import Data.Functor (($>))
import qualified Data.HashMap.Strict as HashMap (elems, intersectionWith)
import Data.List (sortBy)
import qualified Data.Text as Text
import Language.SQL.SpiderSQL.Syntax (SpiderTyp (..))
import Picard.Types (ColumnType (..), SQLSchema (..))
import Text.Parser.Char (CharParsing (..), alphaNum, digit)
import Text.Parser.Combinators (Parsing (notFollowedBy, try, unexpected, (<?>)), between, choice)
import Text.Read (readMaybe)

-- $setup
-- >>> :set -XOverloadedStrings
-- >>> import Data.Attoparsec.Text (parseOnly, endOfInput)

-- | Like 'many' but with an upper limit.
-- @manyAtMost n p@ parses @p@ zero to @n@ times.
manyAtMost :: forall m a. Parsing m => Int -> m a -> m [a]
manyAtMost n q = manyAtMost_q n
  where
    manyAtMost_q 0 = pure []
    manyAtMost_q m = try (someAtMost_q m) <|> pure []
    someAtMost_q 0 = pure []
    someAtMost_q m = liftA2 (:) q (manyAtMost_q (m - 1))

-- | @intP n@ parses an 'Int' with at most @n@ characters.
--
-- >>> parseOnly (intP 3 <* endOfInput) "123"
-- Right 123
--
-- >>> parseOnly (intP 2 <* endOfInput) "123"
-- Left "endOfInput"
intP :: (CharParsing m, Monad m) => Int -> m Int
intP n =
  let q = digit
      p = liftA2 (:) q (manyAtMost (n - 1) q)
   in flip (<?>) "int" $ p >>= maybe empty pure . readMaybe

-- | @doubleP n@ parses a 'Double' with at most @n@ characters.
--
-- >>> parseOnly (doubleP 3 <* endOfInput) "1.2"
-- Right 1.2
--
-- >>> parseOnly (doubleP 2 <* endOfInput) "1.2"
-- Left "double: Failed reading: empty"
--
-- >>> parseOnly (doubleP 8 <* endOfInput) "-1.2e-5"
-- Right (-1.2e-5)
doubleP :: (CharParsing m, Monad m) => Int -> m Double
doubleP n =
  let q = try digit <|> try (char '.') <|> try (char '-') <|> try (char '+') <|> try (char 'e') <|> try (char 'E')
      p = liftA2 (:) q (manyAtMost (n - 1) q)
   in flip (<?>) "double" $ p >>= maybe empty pure . readMaybe

-- | @eitherP p p'@ combines the two alternatives @p@ and @p'@.
eitherP :: forall m a b. Parsing m => m a -> m b -> m (Either a b)
eitherP p p' = try (Left <$> p) <|> try (Right <$> p')

-- | @combine p p'@ merges the results of @p@ and @p'@ using the 'Semigroup' instance.
combine :: (Applicative f, Semigroup a) => f a -> f a -> f a
combine = liftA2 (<>)

-- | @combines ps@ merges the results of the parsers @ps@ using the 'Monoid' instance.
combines :: (Applicative f, Monoid a, Foldable t) => t (f a) -> f a
combines = foldl combine (pure mempty)

-- | @caselessString s@ matches the string @s@ using case insensitive comparison.
--
-- >>> parseOnly (caselessString "singer_in_concert" <* endOfInput) "singer_in_concert"
-- Right "singer_in_concert"
caselessString :: CharParsing m => String -> m String
caselessString = traverse (try . satisfy . ((. toLower) . (==) . toLower))

-- | @keyword k@ is a parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. The parser is not sensitive to
-- letter casing.
--
-- >>> parseOnly (isKeyword "mykeyword" <* endOfInput) "MYKEYWORD"
-- Right "MYKEYWORD"
--
-- >>> parseOnly (isKeyword "mykeyword" <* endOfInput) "MYKEYWRD"
-- Left "mykeyword: satisfyElem"
isKeyword :: CharParsing m => String -> m String
isKeyword s = caselessString s <* notFollowedBy (try alphaNum <|> try (char '_')) <?> s

-- >>> parseOnly (isSelect <* endOfInput) "sEleCt"
-- Right "sEleCt"
--
-- >>> parseOnly (isSelect <* endOfInput) "xelect"
-- Left "select: satisfyElem"
--
-- >>> parseOnly (isSelect <* char 'x' <* endOfInput) "selectx"
-- Left "select: Failed reading: 'x'"
isSelect :: CharParsing m => m String
isSelect = isKeyword "select"

isDistinct :: CharParsing m => m String
isDistinct = isKeyword "distinct"

isStar :: CharParsing m => m String
isStar = pure <$> char '*'

isComma :: CharParsing m => m String
isComma = pure <$> char ','

isDot :: CharParsing m => m String
isDot = pure <$> char '.'

isOpenParenthesis :: CharParsing m => m String
isOpenParenthesis = pure <$> char '('

isClosedParenthesis :: CharParsing m => m String
isClosedParenthesis = pure <$> char ')'

isSingleQuote :: CharParsing m => m String
isSingleQuote = pure <$> char '\''

isDoubleQuote :: CharParsing m => m String
isDoubleQuote = pure <$> char '"'

isEq :: CharParsing m => m String
isEq = pure <$> char '='

isGt :: CharParsing m => m String
isGt = pure <$> char '>'

isLt :: CharParsing m => m String
isLt = pure <$> char '<'

isGe :: CharParsing m => m String
isGe = string ">="

isLe :: CharParsing m => m String
isLe = string "<="

isNe :: CharParsing m => m String
isNe = string "!="

isIn :: CharParsing m => m String
isIn = isKeyword "in"

isLike :: CharParsing m => m String
isLike = isKeyword "like"

isBetween :: CharParsing m => m String
isBetween = isKeyword "between"

isAnd :: CharParsing m => m String
isAnd = isKeyword "and"

isOr :: CharParsing m => m String
isOr = isKeyword "or"

isNot :: CharParsing m => m String
isNot = isKeyword "not"

isMinus :: CharParsing m => m String
isMinus = string "-"

isPlus :: CharParsing m => m String
isPlus = string "+"

isTimes :: CharParsing m => m String
isTimes = string "*"

isDivide :: CharParsing m => m String
isDivide = string "/"

isMax :: CharParsing m => m String
isMax = isKeyword "max"

isMin :: CharParsing m => m String
isMin = isKeyword "min"

isCount :: CharParsing m => m String
isCount = isKeyword "count"

isSum :: CharParsing m => m String
isSum = isKeyword "sum"

isAvg :: CharParsing m => m String
isAvg = isKeyword "avg"

isFrom :: CharParsing m => m String
isFrom = isKeyword "from"

isJoin :: CharParsing m => m String
isJoin = isKeyword "join"

isAs :: CharParsing m => m String
isAs = isKeyword "as"

isOn :: CharParsing m => m String
isOn = isKeyword "on"

isWhere :: CharParsing m => m String
isWhere = isKeyword "where"

isGroupBy :: CharParsing m => m String
isGroupBy = isKeyword "group by"

isOrderBy :: CharParsing m => m String
isOrderBy = isKeyword "order by"

isAsc :: CharParsing m => m String
isAsc = isKeyword "asc"

isDesc :: CharParsing m => m String
isDesc = isKeyword "desc"

isHaving :: CharParsing m => m String
isHaving = isKeyword "having"

isLimit :: CharParsing m => m String
isLimit = isKeyword "limit"

isIntersect :: CharParsing m => m String
isIntersect = isKeyword "intersect"

isExcept :: CharParsing m => m String
isExcept = isKeyword "except"

isUnion :: CharParsing m => m String
isUnion = isKeyword "union"

tableNameP :: (CharParsing m, MonadReader SQLSchema m) => m String
tableNameP = do
  SQLSchema {..} <- ask
  let p cn = caselessString cn $> cn
  choice $
    fmap
      (try . p . Text.unpack)
      (sortBy (compare `on` (negate . Text.length)) (HashMap.elems sQLSchema_tableNames))

columnNameP :: (CharParsing m, MonadReader SQLSchema m) => m String
columnNameP = do
  SQLSchema {..} <- ask
  let p cn = caselessString cn $> cn
  choice $
    fmap
      (try . p . Text.unpack)
      (sortBy (compare `on` (negate . Text.length)) (HashMap.elems sQLSchema_columnNames))

toSpiderTyp :: forall m. Parsing m => ColumnType -> m SpiderTyp
toSpiderTyp ColumnType_BOOLEAN = pure TBoolean
toSpiderTyp ColumnType_TEXT = pure TText
toSpiderTyp ColumnType_NUMBER = pure TNumber
toSpiderTyp ColumnType_TIME = pure TTime
toSpiderTyp ColumnType_OTHERS = pure TOthers
toSpiderTyp (ColumnType__UNKNOWN x) = unexpected $ "unexpected column type " <> show x

toColumnType :: forall m. Parsing m => SpiderTyp -> m ColumnType
toColumnType TBoolean = pure ColumnType_BOOLEAN
toColumnType TText = pure ColumnType_TEXT
toColumnType TNumber = pure ColumnType_NUMBER
toColumnType TTime = pure ColumnType_TIME
toColumnType TOthers = pure ColumnType_OTHERS
toColumnType TStar = unexpected $ "unexpected type " <> show TStar

columnTypeAndNameP :: (CharParsing m, MonadReader SQLSchema m) => m (SpiderTyp, String)
columnTypeAndNameP = do
  SQLSchema {..} <- ask
  let p columnTyp cn = do
        typ <- toSpiderTyp columnTyp
        caselessString cn $> (typ, cn)
  choice $
    fmap
      (\(columnTyp, columnName) -> try (p columnTyp (Text.unpack columnName)))
      ( sortBy
              (compare `on` (negate . Text.length . snd))
              (HashMap.elems $ HashMap.intersectionWith (,) sQLSchema_columnTypes sQLSchema_columnNames)
      )

-- | @quotedString n@ parses a quoted string with at most @n@ characters.
--
-- >>> parseOnly (quotedString 15 <* endOfInput) "\"hello world\""
-- Right "hello world"
--
-- >>> parseOnly (quotedString 10 <* endOfInput) "\"hello world\""
-- Left "quotedString > \"\\\"\": satisfyElem"
--
-- >>> parseOnly (quotedString 15 <* endOfInput) "'hello world'"
-- Right "hello world"
--
-- >>> parseOnly (quotedString 15 <* endOfInput) "'hello world\""
-- Left "quotedString > \"\\\"\": satisfyElem"
--
-- >>> parseOnly (quotedString 15 <* endOfInput) "\"hello world'"
-- Left "quotedString > \"\\\"\": not enough input"
quotedString :: CharParsing m => Int -> m String
quotedString n =
  flip (<?>) "quotedString" $
    try (between (try isSingleQuote) (try isSingleQuote) (try $ manyAtMost n (notChar '\'')))
      <|> try (between (try isDoubleQuote) (try isDoubleQuote) (try $ manyAtMost n (notChar '"')))
