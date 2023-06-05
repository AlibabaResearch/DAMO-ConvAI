module Language.SQL.SpiderSQL.Lexer where

import Control.Applicative (Alternative ((<|>)), Applicative (liftA2))
import Control.Monad.Reader.Class (MonadReader)
import Data.Char (toLower)
import Data.Foldable (asum)
import Language.SQL.SpiderSQL.Prelude (columnNameP, doubleP, intP, isAnd, isAs, isAsc, isAvg, isBetween, isClosedParenthesis, isComma, isCount, isDesc, isDistinct, isDivide, isDot, isEq, isExcept, isFrom, isGe, isGroupBy, isGt, isHaving, isIn, isIntersect, isJoin, isLe, isLike, isLimit, isLt, isMax, isMin, isMinus, isNe, isNot, isOn, isOpenParenthesis, isOr, isOrderBy, isPlus, isSelect, isStar, isSum, isTimes, isUnion, isWhere, manyAtMost, quotedString, tableNameP)
import Picard.Types (SQLSchema (..))
import Text.Parser.Char (CharParsing (..), alphaNum, digit, spaces)
import Text.Parser.Combinators (Parsing (notFollowedBy), sepBy)

-- | @lexSpiderSQL@ produces a list of strings.
--
-- Aliases are restricted to the pattern 'T*' where '*' is equal to one or more digits.
lexSpiderSQL :: (CharParsing m, Monad m, MonadReader SQLSchema m) => m [String]
lexSpiderSQL =
  let keywords = [isSelect, isDistinct, isMax, isMin, isCount, isSum, isAvg, isFrom, isJoin, isOn, isAs, isAnd, isOr, isNot, isIn, isLike, isBetween, isWhere, isGroupBy, isOrderBy, isAsc, isDesc, isHaving, isLimit, isIntersect, isExcept, isUnion]
      punctuation = [isComma, isClosedParenthesis, isOpenParenthesis, isDot]
      operators = [isMinus, isPlus, isTimes, isDivide, isStar, isEq, isGe, isLe, isGt, isLt, isNe]
      primitives = [show <$> doubleP 16, show <$> intP 8, quotedString 32]
      aliasP = do
        _ <- satisfy (\c -> toLower c == 't')
        digits <- liftA2 (:) digit (manyAtMost (9 :: Int) digit)
        pure $ "T" <> digits
      identifiers =
        let terminate q = q <* notFollowedBy (alphaNum <|> char '_')
         in terminate <$> [tableNameP, columnNameP, aliasP]
      p = asum $ identifiers <> keywords <> punctuation <> operators <> primitives
   in sepBy p spaces
