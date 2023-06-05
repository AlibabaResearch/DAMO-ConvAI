{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Main (main, testServer) where

import Control.Applicative (Alternative (empty, many, (<|>)), optional)
import Control.Concurrent (threadDelay)
import Control.Concurrent.Async (mapConcurrently)
import qualified Control.Concurrent.MSem as MSem
import Control.Concurrent.STM.TVar (TVar, modifyTVar, newTVar, readTVar, readTVarIO, writeTVar)
import Control.Exception (throw)
import Control.Monad (ap, forever, void)
import Control.Monad.Except (MonadError (throwError), runExceptT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Reader (runReaderT)
import Control.Monad.STM (STM, atomically, throwSTM)
import Control.Monad.State.Strict (MonadState (get), evalStateT, modify)
import Control.Monad.Trans (MonadTrans (lift))
import Control.Monad.Trans.Free (FreeT)
import qualified Control.Monad.Yoctoparsec as Yocto
import qualified Control.Monad.Yoctoparsec.Class as Yocto
import qualified Data.Attoparsec.Text as Atto (IResult (..), Parser, Result, feed, parse)
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as LBS
import Data.Foldable (Foldable (foldl'))
import Data.Function (on)
import Data.Functor (($>), (<&>))
import qualified Data.HashMap.Strict as HashMap
import Data.Kind (Constraint, Type)
import Data.List (sortBy)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as Text (length, null, pack, stripPrefix, unpack)
import qualified Data.Text.Encoding as Text
import Language.SQL.SpiderSQL.Lexer (lexSpiderSQL)
import Language.SQL.SpiderSQL.Parse (ParserEnv (..), ParserEnvWithGuards (..), mkParserStateTC, mkParserStateUD, spiderSQL, withGuards)
import Language.SQL.SpiderSQL.Prelude (caselessString)
import Language.SQL.SpiderSQL.Syntax (SX (..), SpiderSQLTC, SpiderSQLUD)
import qualified Network.HTTP.Client as HTTP
import qualified Network.HTTP.Client.TLS as HTTP
import qualified Picard.Picard.Client as Picard
import qualified Picard.Picard.Service as Picard
import qualified Picard.Types as Picard
import System.Timeout (timeout)
import Text.Parser.Char (CharParsing (char, string), spaces)
import Text.Parser.Combinators (Parsing (eof))
import qualified Thrift.Api as Thrift
import qualified Thrift.Channel.HeaderChannel as Thrift
import qualified Thrift.Protocol.Id as Thrift
import qualified Thrift.Server.CppServer as Thrift
import qualified Tokenizers (Tokenizer, createTokenizerFromJSONConfig, decode, freeTokenizer)
import Util.Control.Exception (catchAll)
import qualified Util.EventBase as Thrift

trace :: forall a. String -> a -> a
trace _ = id

type RunParsing :: (Type -> Type) -> Type -> Constraint
class CharParsing m => RunParsing m a where
  type Result m a = r | r -> m a
  runParser :: m a -> Result m a
  feed :: Result m a -> Text -> Result m a
  finalize :: Result m a -> Result m a
  toFeedResult :: Result m a -> Picard.FeedResult

instance RunParsing Atto.Parser a where
  type Result Atto.Parser a = Atto.Result a
  runParser p = Atto.parse p mempty
  feed = Atto.feed
  finalize r = case Atto.feed r mempty of
    Atto.Done notConsumed _ | not (Text.null notConsumed) -> Atto.Fail mempty mempty "Not consumed: notConsumed"
    r' -> r'
  toFeedResult (Atto.Done notConsumed _) = Picard.FeedResult_feedCompleteSuccess (Picard.FeedCompleteSuccess notConsumed)
  toFeedResult (Atto.Partial _) = Picard.FeedResult_feedPartialSuccess Picard.FeedPartialSuccess
  toFeedResult (Atto.Fail i contexts description) =
    Picard.FeedResult_feedParseFailure
      Picard.FeedParseFailure
        { feedParseFailure_input = i,
          feedParseFailure_contexts = Text.pack <$> contexts,
          feedParseFailure_description = Text.pack description
        }

instance RunParsing (FreeT ((->) Char) []) a where
  type Result (FreeT ((->) Char) []) a = [Yocto.Result [] Char a]
  runParser p = Yocto.runParser p
  feed r s = do
    r' <- r
    Yocto.feed r' . Text.unpack $ s
  finalize =
    foldMap
      ( \case
          (Yocto.Done a []) -> pure $ Yocto.Done a []
          (Yocto.Done _ _) -> mempty
          (Yocto.Partial _) -> mempty
      )
  toFeedResult [] =
    Picard.FeedResult_feedParseFailure
      Picard.FeedParseFailure
        { feedParseFailure_input = mempty,
          feedParseFailure_contexts = mempty,
          feedParseFailure_description = "Nothing remains"
        }
  toFeedResult (Yocto.Done _ notConsumed : _) = Picard.FeedResult_feedCompleteSuccess . Picard.FeedCompleteSuccess $ Text.pack notConsumed
  toFeedResult (Yocto.Partial _ : _) = Picard.FeedResult_feedPartialSuccess Picard.FeedPartialSuccess

data PartialParse m a = PartialParse !Text !(Result m a)

deriving stock instance (Show (Result m a)) => Show (PartialParse m a)

type Detokenize = Picard.InputIds -> IO String

data PicardState = PicardState
  { psCounter :: TVar Int,
    psSQLSchemas :: TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema),
    psTokenizer :: TVar (Maybe Tokenizers.Tokenizer),
    psDetokenize :: TVar (Maybe Detokenize),
    psPartialSpiderSQLParsesWithGuardsAndTypeChecking ::
      TVar
        ( HashMap.HashMap
            Picard.InputIds
            (PartialParse (FreeT ((->) Char) []) SpiderSQLTC)
        ),
    psPartialSpiderSQLParsesWithGuards ::
      TVar
        ( HashMap.HashMap
            Picard.InputIds
            (PartialParse Atto.Parser SpiderSQLUD)
        ),
    psPartialSpiderSQLParsesWithoutGuards ::
      TVar
        ( HashMap.HashMap
            Picard.InputIds
            (PartialParse Atto.Parser SpiderSQLUD)
        ),
    psPartialSpiderSQLLexes ::
      TVar
        ( HashMap.HashMap
            Picard.InputIds
            (PartialParse Atto.Parser [String])
        )
  }

initPicardState :: IO PicardState
initPicardState =
  atomically $
    PicardState
      <$> newTVar 0
        <*> newTVar mempty
        <*> newTVar Nothing
        <*> newTVar Nothing
        <*> newTVar mempty
        <*> newTVar mempty
        <*> newTVar mempty
        <*> newTVar mempty

mkSchemaParser ::
  forall m.
  CharParsing m =>
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  m Picard.SQLSchema
mkSchemaParser sqlSchemas =
  foldl'
    (\agg (dbId, schema) -> agg <|> caselessString (Text.unpack dbId) $> schema)
    empty
    (sortBy (compare `on` (negate . Text.length . fst)) (HashMap.toList sqlSchemas))

mkParser ::
  forall m a.
  (Monad m, CharParsing m) =>
  m Picard.SQLSchema ->
  (Picard.SQLSchema -> m a) ->
  m a
mkParser schemaParser mkMainParser = do
  _ <-
    spaces
      *> many
        ( char '<'
            *> (string "pad" <|> string "s" <|> string "/s")
            <* char '>'
        )
        <* spaces
  schema <- schemaParser
  _ <- spaces *> char '|' <* spaces
  mkMainParser schema
    <* optional (spaces <* char ';')
    <* eof

getPartialParse ::
  forall m a.
  (RunParsing m a, Monad m) =>
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  (Picard.SQLSchema -> m a) ->
  Text ->
  PartialParse m a
getPartialParse sqlSchemas mkMainParser =
  let schemaParser = mkSchemaParser sqlSchemas
      m = mkParser schemaParser mkMainParser
   in ap PartialParse $ feed (runParser m)

initializeParserCacheSTM ::
  forall m a.
  (RunParsing m a, Monad m) =>
  (Picard.SQLSchema -> m a) ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  STM ()
initializeParserCacheSTM mainParser sqlSchemas partialParses = do
  nukeParserCache partialParses
  partialParse <-
    getPartialParse
      <$> readTVar sqlSchemas
      <*> pure mainParser
      <*> pure mempty
  modifyTVar partialParses (HashMap.insert mempty partialParse)

nukeParserCache ::
  forall m a.
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  STM ()
nukeParserCache partialParses = writeTVar partialParses HashMap.empty

data LookupResult a = Cached !a | Fresh !a
  deriving stock (Show)

lookupResultIO ::
  forall m a.
  (RunParsing m a, Monad m) =>
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  (Picard.SQLSchema -> m a) ->
  (Picard.InputIds -> IO String) ->
  HashMap.HashMap Picard.InputIds (PartialParse m a) ->
  Picard.InputIds ->
  IO (LookupResult (PartialParse m a))
lookupResultIO sqlSchemas mkMainParser decode partialParses inputIds =
  case HashMap.lookup inputIds partialParses of
    Just partialParse ->
      trace ("Server: Found inputIds " <> show inputIds) . pure $ Cached partialParse
    Nothing ->
      trace ("Server: Did not find inputIds " <> show inputIds) $ do
        decodedInputIds <- decode inputIds
        let !partialParse = getPartialParse sqlSchemas mkMainParser (Text.pack decodedInputIds)
        pure $ Fresh partialParse

lookupResultWithTimeoutIO ::
  forall m a n.
  (RunParsing m a, Monad m, MonadState DebugInfo n, MonadError Picard.FeedTimeoutFailure n, MonadIO n) =>
  Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> m a) ->
  (Picard.InputIds -> IO String) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  Picard.InputIds ->
  n (PartialParse m a)
lookupResultWithTimeoutIO microSeconds sqlSchemas mkMainParser decode partialParses inputIds =
  resultOrTimeout
    microSeconds
    ( do
        schemas <- readTVarIO sqlSchemas
        parses <- readTVarIO partialParses
        !lr <- lookupResultIO schemas mkMainParser decode parses inputIds
        pure lr
    )
    >>= cache
  where
    cache (Cached partialParse) = pure partialParse
    cache (Fresh partialParse) =
      trace ("Server: Cached inputIds " <> show inputIds) . liftIO . atomically $ do
        modifyTVar partialParses $ HashMap.insert inputIds partialParse
        pure partialParse

decodedTokenFromDifferenceIO ::
  (Picard.InputIds -> IO String) ->
  Picard.InputIds ->
  Picard.Token ->
  Text ->
  IO (Text, Maybe Text)
decodedTokenFromDifferenceIO decode inputIds token decodedInputIds = do
  decoded <- Text.pack <$> decode (inputIds ++ [token])
  pure (decoded, Text.stripPrefix decodedInputIds decoded)

decodedTokenFromDifferenceM ::
  forall m.
  (MonadState DebugInfo m, MonadIO m) =>
  (Picard.InputIds -> IO String) ->
  Picard.InputIds ->
  Picard.Token ->
  Text ->
  m (Text, Text)
decodedTokenFromDifferenceM decode inputIds token decodedInputIds = do
  (decoded, maybeDecodedToken) <- liftIO $ decodedTokenFromDifferenceIO decode inputIds token decodedInputIds
  _ <- modify (\debugInfo -> debugInfo {debugDecoded = Just decoded})
  maybe
    ( trace
        ("Server: Prefix error " <> show decodedInputIds <> " " <> show decoded)
        . throw
        . Picard.FeedException
        . Picard.FeedFatalException_tokenizerPrefixException
        . Picard.TokenizerPrefixException
        $ "Prefix error."
    )
    ( \decodedToken -> do
        _ <- modify (\debugInfo -> debugInfo {debugDecodedToken = Just decodedToken})
        pure (decoded, decodedToken)
    )
    maybeDecodedToken

data DebugInfo = DebugInfo
  { debugInputIds :: Maybe Picard.InputIds,
    debugToken :: Maybe Picard.Token,
    debugDecodedInputIds :: Maybe Text,
    debugDecodedToken :: Maybe Text,
    debugDecoded :: Maybe Text
  }

mkDebugInfo :: DebugInfo
mkDebugInfo = DebugInfo Nothing Nothing Nothing Nothing Nothing

resultOrTimeout ::
  forall r m.
  (MonadIO m, MonadState DebugInfo m, MonadError Picard.FeedTimeoutFailure m) =>
  Int ->
  IO r ->
  m r
resultOrTimeout microSeconds ior = do
  mr <- liftIO $ timeout microSeconds ior
  case mr of
    Just r -> pure r
    Nothing -> do
      DebugInfo {..} <- get
      trace
        ("Server: Timeout error " <> show debugDecodedInputIds <> " " <> show debugDecoded)
        . throwError
        . Picard.FeedTimeoutFailure
        $ "Timeout error."

feedParserWithTimeoutIO ::
  forall m a n.
  (RunParsing m a, MonadState DebugInfo n, MonadIO n, MonadError Picard.FeedTimeoutFailure n) =>
  Int ->
  Result m a ->
  Text ->
  n (Result m a)
feedParserWithTimeoutIO microSeconds partialParseResult decodedToken = do
  resultOrTimeout
    microSeconds
    $ let !r = case decodedToken of
            "</s>" -> finalize partialParseResult
            s -> feed partialParseResult s
       in pure r

-- | fix me: need one counter for each hashmap
nukeParserCacheEverySTM ::
  forall m a.
  Int ->
  TVar Int ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  STM ()
nukeParserCacheEverySTM n counter partialParses = do
  _ <- modifyTVar counter (+ 1)
  c <- readTVar counter
  case c `mod` n of
    0 -> nukeParserCache partialParses
    _ -> pure ()

toFeedResult' ::
  forall m a n.
  (RunParsing m a, MonadState DebugInfo n, Show (Result m a)) =>
  Result m a ->
  n Picard.FeedResult
toFeedResult' r = do
  let fr = toFeedResult r
  DebugInfo {..} <- get
  pure . flip trace fr $
    "Server: "
      <> ( case fr of
             Picard.FeedResult_feedParseFailure Picard.FeedParseFailure {} -> "Failure"
             Picard.FeedResult_feedTimeoutFailure (Picard.FeedTimeoutFailure msg) -> "Timeout failure: " <> Text.unpack msg
             Picard.FeedResult_feedPartialSuccess Picard.FeedPartialSuccess -> "Partial"
             Picard.FeedResult_feedCompleteSuccess (Picard.FeedCompleteSuccess _) -> "Success"
             Picard.FeedResult_EMPTY -> "Unknown"
         )
      <> ". Input ids were: "
      <> show debugInputIds
      <> ". Token was: "
      <> show debugToken
      <> ". Decoded input ids were: "
      <> show debugDecodedInputIds
      <> ". Decoded token was: "
      <> show debugDecodedToken
      <> ". Result: "
      <> show r
      <> "."

getDetokenize :: TVar (Maybe Detokenize) -> IO Detokenize
getDetokenize =
  fmap
    ( fromMaybe
        ( throw
            . Picard.FeedException
            . Picard.FeedFatalException_tokenizerNotRegisteredException
            . Picard.TokenizerNotRegisteredException
            $ "Tokenizer has not been registered."
        )
    )
    . readTVarIO

feedIO ::
  forall m a.
  (RunParsing m a, Monad m, Show (Result m a)) =>
  Int ->
  TVar Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> m a) ->
  Detokenize ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  Picard.InputIds ->
  Picard.Token ->
  IO Picard.FeedResult
feedIO microSeconds counter sqlSchemas mkMainParser detokenize partialParses inputIds token =
  evalStateT
    ( runExceptT
        ( do
            _ <- liftIO . atomically $ nukeParserCacheEverySTM 10000 counter partialParses
            partialParse <- getPartialParseIO detokenize
            liftIO . atomically . modifyTVar partialParses $ HashMap.insert (inputIds ++ [token]) partialParse
            pure partialParse
        )
        >>= toFeedResultIO
    )
    initialDebugInfo
  where
    initialDebugInfo =
      mkDebugInfo
        { debugInputIds = Just inputIds,
          debugToken = Just token
        }
    getPartialParseIO tokenizer = do
      PartialParse decodedInputIds partialParseResult <-
        lookupResultWithTimeoutIO microSeconds sqlSchemas mkMainParser tokenizer partialParses inputIds
      (decoded, decodedToken) <- decodedTokenFromDifferenceM tokenizer inputIds token decodedInputIds
      modify (\debugInfo -> debugInfo {debugDecodedInputIds = Just decodedInputIds})
      partialParseResult' <- feedParserWithTimeoutIO microSeconds partialParseResult decodedToken
      pure $ PartialParse decoded partialParseResult'
    toFeedResultIO (Left timeoutFailure) = pure $ Picard.FeedResult_feedTimeoutFailure timeoutFailure
    toFeedResultIO (Right (PartialParse _ r)) = toFeedResult' r

batchFeedIO ::
  forall m a.
  (RunParsing m a, Monad m, Show (Result m a)) =>
  Int ->
  TVar Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> m a) ->
  Detokenize ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse m a)) ->
  [Picard.InputIds] ->
  [[Picard.Token]] ->
  IO [Picard.BatchFeedResult]
batchFeedIO microSeconds counter sqlSchemas mkMainParser detokenize partialParses inputIds topTokens =
  do
    -- traverse
    -- mapPool (length inputIds)
    mapConcurrently
      ( \(batchId, inputIds', token) ->
          feedIO microSeconds counter sqlSchemas mkMainParser detokenize partialParses inputIds' token
            <&> Picard.BatchFeedResult batchId token
      )
      . concat
      . zipWith3
        (\batchId inputIds' tokens -> (batchId,inputIds',) <$> tokens)
        [0 :: Picard.BatchId ..]
        inputIds
      $ topTokens

mapPool :: forall a b t. Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool size f xs = do
  sem <- MSem.new size
  mapConcurrently (MSem.with sem . f) xs

picardHandler :: forall a. PicardState -> Picard.PicardCommand a -> IO a
picardHandler PicardState {..} = go
  where
    mkSpiderSQLParserWithGuardsAndTypeChecking :: Picard.SQLSchema -> Yocto.Parser [] Char SpiderSQLTC
    mkSpiderSQLParserWithGuardsAndTypeChecking = runReaderT (spiderSQL STC mkParserStateTC) . ParserEnv (ParserEnvWithGuards (withGuards STC))
    mkSpiderSQLParserWithGuards :: Picard.SQLSchema -> Atto.Parser SpiderSQLUD
    mkSpiderSQLParserWithGuards = runReaderT (spiderSQL SUD mkParserStateUD) . ParserEnv (ParserEnvWithGuards (withGuards SUD))
    mkSpiderSQLParserWithoutGuards :: Picard.SQLSchema -> Atto.Parser SpiderSQLUD
    mkSpiderSQLParserWithoutGuards = runReaderT (spiderSQL SUD mkParserStateUD) . ParserEnv (ParserEnvWithGuards (const id))
    mkSpiderSQLLexer :: Picard.SQLSchema -> Atto.Parser [String]
    mkSpiderSQLLexer = runReaderT lexSpiderSQL
    go (Picard.RegisterSQLSchema dbId sqlSchema) =
      trace ("RegisterSQLSchema " <> show dbId) $
        atomically $ do
          r <- readTVar psSQLSchemas
          case HashMap.lookup dbId r of
            Just _ -> throwSTM $ Picard.RegisterSQLSchemaException dbId "Database schema is already registered"
            Nothing -> do
              modifyTVar psSQLSchemas (HashMap.insert dbId sqlSchema)
              initializeParserCacheSTM mkSpiderSQLParserWithGuardsAndTypeChecking psSQLSchemas psPartialSpiderSQLParsesWithGuardsAndTypeChecking
              initializeParserCacheSTM mkSpiderSQLParserWithGuards psSQLSchemas psPartialSpiderSQLParsesWithGuards
              initializeParserCacheSTM mkSpiderSQLParserWithoutGuards psSQLSchemas psPartialSpiderSQLParsesWithoutGuards
              initializeParserCacheSTM mkSpiderSQLLexer psSQLSchemas psPartialSpiderSQLLexes
    go (Picard.RegisterTokenizer jsonConfig) =
      trace "RegisterTokenizer" $ do
        tok <- Tokenizers.createTokenizerFromJSONConfig . Text.encodeUtf8 $ jsonConfig
        tokSem <- MSem.new (1 :: Int)
        maybeOldTokenizer <- atomically $ do
          maybeOldTokenizer <- readTVar psTokenizer
          writeTVar psTokenizer . Just $ tok
          writeTVar psDetokenize . Just $ \inputIds -> MSem.with tokSem (Tokenizers.decode tok $ fromIntegral <$> inputIds)
          initializeParserCacheSTM mkSpiderSQLParserWithGuardsAndTypeChecking psSQLSchemas psPartialSpiderSQLParsesWithGuardsAndTypeChecking
          initializeParserCacheSTM mkSpiderSQLParserWithGuards psSQLSchemas psPartialSpiderSQLParsesWithGuards
          initializeParserCacheSTM mkSpiderSQLParserWithoutGuards psSQLSchemas psPartialSpiderSQLParsesWithoutGuards
          initializeParserCacheSTM mkSpiderSQLLexer psSQLSchemas psPartialSpiderSQLLexes
          pure maybeOldTokenizer
        case maybeOldTokenizer of
          Just oldTok -> Tokenizers.freeTokenizer oldTok
          Nothing -> pure ()
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITH_GUARDS_AND_TYPE_CHECKING) =
      trace ("Feed parsing with guards " <> show inputIds <> " " <> show token) $ do
        detokenize <- getDetokenize psDetokenize
        feedIO 100000 psCounter psSQLSchemas mkSpiderSQLParserWithGuardsAndTypeChecking detokenize psPartialSpiderSQLParsesWithGuardsAndTypeChecking inputIds token
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITH_GUARDS) =
      trace ("Feed parsing with guards " <> show inputIds <> " " <> show token) $ do
        detokenize <- getDetokenize psDetokenize
        feedIO 100000 psCounter psSQLSchemas mkSpiderSQLParserWithGuards detokenize psPartialSpiderSQLParsesWithGuards inputIds token
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITHOUT_GUARDS) =
      trace ("Feed parsing without guards " <> show inputIds <> " " <> show token) $ do
        detokenize <- getDetokenize psDetokenize
        feedIO 100000 psCounter psSQLSchemas mkSpiderSQLParserWithoutGuards detokenize psPartialSpiderSQLParsesWithoutGuards inputIds token
    go (Picard.Feed inputIds token Picard.Mode_LEXING) =
      trace ("Feed lexing " <> show inputIds <> " " <> show token) $ do
        detokenize <- getDetokenize psDetokenize
        feedIO 100000 psCounter psSQLSchemas mkSpiderSQLLexer detokenize psPartialSpiderSQLLexes inputIds token
    go (Picard.Feed _inputIds _token (Picard.Mode__UNKNOWN n)) =
      throw
        . Picard.FeedException
        . Picard.FeedFatalException_modeException
        . Picard.ModeException
        . Text.pack
        $ "Unknown mode " <> show n
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITH_GUARDS_AND_TYPE_CHECKING) = do
      detokenize <- getDetokenize psDetokenize
      batchFeedIO 10000000 psCounter psSQLSchemas mkSpiderSQLParserWithGuardsAndTypeChecking detokenize psPartialSpiderSQLParsesWithGuardsAndTypeChecking inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITH_GUARDS) = do
      detokenize <- getDetokenize psDetokenize
      batchFeedIO 10000000 psCounter psSQLSchemas mkSpiderSQLParserWithGuards detokenize psPartialSpiderSQLParsesWithGuards inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITHOUT_GUARDS) = do
      detokenize <- getDetokenize psDetokenize
      batchFeedIO 10000000 psCounter psSQLSchemas mkSpiderSQLParserWithoutGuards detokenize psPartialSpiderSQLParsesWithoutGuards inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_LEXING) = do
      detokenize <- getDetokenize psDetokenize
      batchFeedIO 10000000 psCounter psSQLSchemas mkSpiderSQLLexer detokenize psPartialSpiderSQLLexes inputIds topTokens
    go (Picard.BatchFeed _inputIds _token (Picard.Mode__UNKNOWN n)) =
      throw
        . Picard.FeedException
        . Picard.FeedFatalException_modeException
        . Picard.ModeException
        . Text.pack
        $ "Unknown mode " <> show n

withPicardServer :: forall a. Thrift.ServerOptions -> (Int -> IO a) -> IO a
withPicardServer serverOptions action = do
  st <- initPicardState
  Thrift.withBackgroundServer (picardHandler st) serverOptions $
    \Thrift.Server {..} -> action serverPort

picardServerHost :: BS8.ByteString
picardServerHost = BS8.pack "127.0.0.1"

mkHeaderConfig :: forall t. Int -> Thrift.ProtocolId -> Thrift.HeaderConfig t
mkHeaderConfig port protId =
  Thrift.HeaderConfig
    { headerHost = picardServerHost,
      headerPort = port,
      headerProtocolId = protId,
      headerConnTimeout = 5000,
      headerSendTimeout = 5000,
      headerRecvTimeout = 5000
    }

testServer :: IO ()
testServer = do
  let protId = Thrift.binaryProtocolId
      action :: Thrift.Thrift Picard.Picard ()
      action = do
        Picard.registerSQLSchema "test" $
          let columnNames = HashMap.fromList [("0", "column")]
              columnTypes = HashMap.fromList [("0", Picard.ColumnType_NUMBER)]
              tableNames = HashMap.fromList [("0", "table")]
              columnToTable = HashMap.fromList [("0", "0")]
              tableToColumns = HashMap.fromList [("0", ["0"])]
              foreignKeys = HashMap.fromList []
              primaryKeys = []
           in Picard.SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}
        Picard.registerSQLSchema "car_1" $
          let columnNames = HashMap.fromList [("1", "ContId"), ("10", "ModelId"), ("11", "Maker"), ("12", "Model"), ("13", "MakeId"), ("14", "Model"), ("15", "Make"), ("16", "Id"), ("17", "MPG"), ("18", "Cylinders"), ("19", "Edispl"), ("2", "Continent"), ("20", "Horsepower"), ("21", "Weight"), ("22", "Accelerate"), ("23", "Year"), ("3", "CountryId"), ("4", "CountryName"), ("5", "Continent"), ("6", "Id"), ("7", "Maker"), ("8", "FullName"), ("9", "Country")]
              columnTypes = HashMap.fromList [("1", Picard.ColumnType_NUMBER), ("10", Picard.ColumnType_NUMBER), ("11", Picard.ColumnType_NUMBER), ("12", Picard.ColumnType_TEXT), ("13", Picard.ColumnType_NUMBER), ("14", Picard.ColumnType_TEXT), ("15", Picard.ColumnType_TEXT), ("16", Picard.ColumnType_NUMBER), ("17", Picard.ColumnType_TEXT), ("18", Picard.ColumnType_NUMBER), ("19", Picard.ColumnType_NUMBER), ("2", Picard.ColumnType_TEXT), ("20", Picard.ColumnType_TEXT), ("21", Picard.ColumnType_NUMBER), ("22", Picard.ColumnType_NUMBER), ("23", Picard.ColumnType_NUMBER), ("3", Picard.ColumnType_NUMBER), ("4", Picard.ColumnType_TEXT), ("5", Picard.ColumnType_NUMBER), ("6", Picard.ColumnType_NUMBER), ("7", Picard.ColumnType_TEXT), ("8", Picard.ColumnType_TEXT), ("9", Picard.ColumnType_TEXT)]
              tableNames = HashMap.fromList [("0", "continents"), ("1", "countries"), ("2", "car_makers"), ("3", "model_list"), ("4", "car_names"), ("5", "cars_data")]
              columnToTable = HashMap.fromList [("1", "0"), ("10", "3"), ("11", "3"), ("12", "3"), ("13", "4"), ("14", "4"), ("15", "4"), ("16", "5"), ("17", "5"), ("18", "5"), ("19", "5"), ("2", "0"), ("20", "5"), ("21", "5"), ("22", "5"), ("23", "5"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "2"), ("7", "2"), ("8", "2"), ("9", "2")]
              tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5"]), ("2", ["6", "7", "8", "9"]), ("3", ["10", "11", "12"]), ("4", ["13", "14", "15"]), ("5", ["16", "17", "18", "19", "20", "21", "22", "23"])]
              foreignKeys = HashMap.fromList [("11", "6"), ("14", "12"), ("16", "13"), ("5", "1"), ("9", "3")]
              primaryKeys = ["1", "3", "6", "10", "13", "16"]
           in Picard.SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}
        manager <- lift $ HTTP.newTlsManagerWith HTTP.tlsManagerSettings
        request <- lift $ HTTP.parseRequest "https://huggingface.co/t5-base/resolve/main/tokenizer.json"
        response <- lift $ HTTP.httpLbs request manager
        Picard.registerTokenizer . Text.decodeUtf8 . LBS.toStrict $ HTTP.responseBody response
        let tokens = [0 .. 32100]
        mapM_
          ( flip catchAll (const (pure Nothing))
              . (Just <$>)
              . (\token -> Picard.feed [0, 794, 1820, 1738, 953, 5, 3297, 440, 29, 45, 953] token Picard.Mode_PARSING_WITH_GUARDS)
          )
          tokens
        let inputIds = [0, 443, 834, 536, 1820, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 563, 57, 3, 17, 5411, 17529, 23, 26, 578, 3476, 599, 1935, 61, 2490, 220, 7021, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 1715, 825, 834, 3350, 38, 3, 17, 519, 30, 3, 17, 4416, 23, 26, 3274, 3, 17, 5787, 8337, 213, 3, 17, 5787, 21770, 3274, 96, 3183, 144, 121, 1]
        void $
          Picard.feed
            (take 127 inputIds)
            (inputIds !! 127)
            Picard.Mode_PARSING_WITH_GUARDS
        void $
          Picard.feed
            (take 128 inputIds)
            (inputIds !! 128)
            Picard.Mode_PARSING_WITH_GUARDS
  withPicardServer Thrift.defaultOptions $
    \port ->
      Thrift.withEventBaseDataplane $ \evb -> do
        let headerConf = mkHeaderConfig port protId
        Thrift.withHeaderChannel evb headerConf action

main :: IO ()
main = do
  st <- initPicardState
  let serverOptions =
        Thrift.ServerOptions
          { desiredPort = Just 9090,
            customFactoryFn = Nothing,
            customModifyFn = Nothing
          }
      action Thrift.Server {} =
        forever (threadDelay maxBound)
  Thrift.withBackgroundServer (picardHandler st) serverOptions action
