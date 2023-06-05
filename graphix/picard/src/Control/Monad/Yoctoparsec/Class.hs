{-# OPTIONS_GHC -Wno-orphans #-}

module Control.Monad.Yoctoparsec.Class where

import Control.Applicative (Alternative (empty))
import Control.Monad (MonadPlus, foldM, mfilter)
import Control.Monad.Trans.Free (FreeF (..), FreeT (..))
import qualified Control.Monad.Yoctoparsec as Yocto
import Data.Foldable (Foldable (..))
import Text.Parser.Char (CharParsing (..))
import Text.Parser.Combinators (Parsing (..))
import Text.Parser.Token (TokenParsing)

instance (MonadPlus b, MonadFail b) => Parsing (FreeT ((->) i) b) where
  try = id
  (<?>) = const
  unexpected = fail
  eof = pure ()
  notFollowedBy _ = pure ()

instance (MonadPlus b, MonadFail b) => CharParsing (FreeT ((->) Char) b) where
  satisfy p = mfilter p Yocto.token

instance (MonadPlus b, MonadFail b) => TokenParsing (FreeT ((->) Char) b)

data Result b i a = Partial (i -> b (Result b i a)) | Done a [i]

instance (Show a, Show i) => Show (Result b i a) where
  show (Partial _) = "Partial"
  show (Done a is) = "Done (" <> show a <> ") (" <> show is <> ")"

runParser :: forall b i a. (Monad b, Foldable b) => Yocto.Parser b i a -> b (Result b i a)
runParser p = do
  val <- runFreeT p
  pure $ case val of
    Pure a -> Done a []
    Free f -> Partial (runParser . f)

feed1 :: forall b i a. Applicative b => Result b i a -> i -> b (Result b i a)
feed1 (Partial f) i = f i
feed1 (Done a is) i = pure $ Done a (is <> [i])

feed :: forall t b i a. (Foldable t, Monad b) => Result b i a -> t i -> b (Result b i a)
feed = foldM feed1

feedOnly :: forall t b i a. (Foldable t, Monad b, Alternative b) => Result b i a -> t i -> b (Result b i a)
feedOnly r t =
  go r (toList t)
  where
    go (Partial _) [] = empty
    go (Done a is) [] = pure $ Done a is
    go r' (i : is) = do
      r'' <- feed1 r' i
      go r'' is
