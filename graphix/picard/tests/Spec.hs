module Main where

import Control.Applicative (Alternative (empty), optional)
import Control.Monad.Reader (runReaderT)
import Control.Monad.Trans (lift)
import qualified Control.Monad.Yoctoparsec.Class as Yocto
import qualified Data.Attoparsec.Text as Atto (parseOnly)
import qualified Data.Text as Text
import Language.SQL.SpiderSQL.Academic (academicLexerTests, academicParserTests)
import Language.SQL.SpiderSQL.AssetsMaintenance (assetsMaintenanceLexerTests, assetsMaintenanceParserTests)
import Language.SQL.SpiderSQL.Bike1 (bike1LexerTests, bike1ParserTests)
import Language.SQL.SpiderSQL.Car1 (car1LexerTests, car1ParserTests)
import Language.SQL.SpiderSQL.Chinook1 (chinook1LexerTests, chinook1ParserTests)
import Language.SQL.SpiderSQL.ConcertSinger (concertSingerLexerTests, concertSingerParserTests)
import Language.SQL.SpiderSQL.CreDocTemplateMgt (creDocTemplateMgtLexerTests, creDocTemplateMgtParserTests)
import Language.SQL.SpiderSQL.DepartmentManagement (departmentManagementLexerTests, departmentManagementParserTests)
import Language.SQL.SpiderSQL.DogKennels (dogKennelsLexerTests, dogKennelsParserTests)
import Language.SQL.SpiderSQL.Flight1 (flight1LexerTests, flight1ParserTests)
import Language.SQL.SpiderSQL.Geo (geoLexerTests, geoParserTests)
import Language.SQL.SpiderSQL.Inn1 (inn1LexerTests, inn1ParserTests)
import Language.SQL.SpiderSQL.Lexer (lexSpiderSQL)
import Language.SQL.SpiderSQL.MatchSeason (matchSeasonLexerTests, matchSeasonParserTests)
import Language.SQL.SpiderSQL.MuseumVisit (museumVisitLexerTests, museumVisitParserTests)
import Language.SQL.SpiderSQL.Orchestra (orchestraLexerTests, orchestraParserTests)
import Language.SQL.SpiderSQL.Parse (ParserEnv (..), ParserEnvWithGuards (..), mkParserStateTC, mkParserStateUD, spiderSQL, withGuards)
import Language.SQL.SpiderSQL.Pets1 (pets1LexerTests, pets1ParserTests)
import Language.SQL.SpiderSQL.PhoneMarket (phoneMarketLexerTests, phoneMarketParserTests)
import Language.SQL.SpiderSQL.PokerPlayer (pokerPlayerLexerTests, pokerPlayerParserTests)
import Language.SQL.SpiderSQL.ProductCatalog (productCatalogLexerTests, productCatalogParserTests)
import Language.SQL.SpiderSQL.Scholar (scholarLexerTests, scholarParserTests)
import Language.SQL.SpiderSQL.Singer (singerLexerTests, singerParserTests)
import Language.SQL.SpiderSQL.StormRecord (stormRecordLexerTests, stormRecordParserTests)
import Language.SQL.SpiderSQL.StudentTranscriptsTracking (studentTranscriptsTrackingLexerTests, studentTranscriptsTrackingParserTests)
import Language.SQL.SpiderSQL.Syntax (SX (..))
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Language.SQL.SpiderSQL.Wta1 (wta1LexerTests, wta1ParserTests)
import qualified Test.Tasty as T
import qualified Test.Tasty.HUnit as H
import Text.Parser.Char (CharParsing (..), spaces)
import Text.Parser.Combinators (Parsing (..))
import qualified Text.Trifecta.Parser as Trifecta
import qualified Text.Trifecta.Result as Trifecta

-- | Run 'cabal repl test:spec' to get a REPL for the tests.
main :: IO ()
main = T.defaultMain testTree

testData :: TestItem
testData =
  Group
    "tests"
    [ Group
        "lexer"
        [ academicLexerTests,
          assetsMaintenanceLexerTests,
          bike1LexerTests,
          car1LexerTests,
          chinook1LexerTests,
          concertSingerLexerTests,
          creDocTemplateMgtLexerTests,
          departmentManagementLexerTests,
          dogKennelsLexerTests,
          flight1LexerTests,
          geoLexerTests,
          inn1LexerTests,
          matchSeasonLexerTests,
          museumVisitLexerTests,
          orchestraLexerTests,
          pets1LexerTests,
          phoneMarketLexerTests,
          pokerPlayerLexerTests,
          productCatalogLexerTests,
          scholarLexerTests,
          singerLexerTests,
          stormRecordLexerTests,
          studentTranscriptsTrackingLexerTests,
          wta1LexerTests
        ],
      Group
        "parser"
        [ academicParserTests,
          assetsMaintenanceParserTests,
          bike1ParserTests,
          car1ParserTests,
          chinook1ParserTests,
          concertSingerParserTests,
          creDocTemplateMgtParserTests,
          departmentManagementParserTests,
          dogKennelsParserTests,
          flight1ParserTests,
          geoParserTests,
          inn1ParserTests,
          matchSeasonParserTests,
          museumVisitParserTests,
          orchestraParserTests,
          pets1ParserTests,
          phoneMarketParserTests,
          pokerPlayerParserTests,
          productCatalogParserTests,
          scholarParserTests,
          singerParserTests,
          stormRecordParserTests,
          studentTranscriptsTrackingParserTests,
          wta1ParserTests
        ]
    ]

testTree :: T.TestTree
testTree = toTest testData
  where
    withEnv parserEnv p =
      runReaderT
        ( p
            <* optional (lift $ spaces <* char ';')
            <* lift eof
        )
        parserEnv
    attoParseOnly = Atto.parseOnly
    trifectaParseOnly p query = Trifecta.parseString p mempty (Text.unpack query)
    yoctoParseOnly p query =
      foldMap @[]
        ( \case
            (Yocto.Done a []) -> pure a
            (Yocto.Done _ _) -> empty
            (Yocto.Partial _) -> empty
        )
        $ do
          p' <- Yocto.runParser p
          Yocto.feedOnly p' (Text.unpack query)
    toTest (Group name tests) =
      T.testGroup name $ toTest <$> tests
    toTest (LexQueryExpr sqlSchema query) =
      H.testCase ("Lex " <> show query) $
        let p = withEnv sqlSchema lexSpiderSQL
         in case attoParseOnly p query of
              Left e -> H.assertFailure e
              Right _ -> pure ()
    toTest (ParseQueryExprWithoutGuards sqlSchema query) =
      H.testCase ("Parse without guards " <> show query) $
        let p = withEnv (ParserEnv (ParserEnvWithGuards (const id)) sqlSchema) (spiderSQL SUD mkParserStateUD)
         in case attoParseOnly p query of
              Left e -> H.assertFailure e
              Right _ -> pure ()
    toTest (ParseQueryExprWithGuards sqlSchema query) =
      H.testCase ("Parse with guards " <> show query) $
        let p = withEnv (ParserEnv (ParserEnvWithGuards (withGuards SUD)) sqlSchema) (spiderSQL SUD mkParserStateUD)
         in -- case yoctoParseOnly p query of
            --   _ : _ -> pure ()
            --   [] -> H.assertFailure "empty"
            -- case attoParseOnly p query of
            --   Left e -> H.assertFailure e
            --   Right _ -> pure ()
            case trifectaParseOnly p query of
              Trifecta.Failure Trifecta.ErrInfo {..} -> H.assertFailure (show _errDoc)
              Trifecta.Success _ -> pure ()
    toTest (ParseQueryExprWithGuardsAndTypeChecking sqlSchema query) =
      H.testCase ("Parse and type check " <> show query) $
        let p = withEnv (ParserEnv (ParserEnvWithGuards (withGuards STC)) sqlSchema) (spiderSQL STC mkParserStateTC)
         in case yoctoParseOnly p query of
              _ : _ -> pure ()
              [] -> H.assertFailure "empty"
    toTest (ParseQueryExprFails sqlSchema query) =
      H.testCase ("Fail " <> show query) $
        let p = withEnv (ParserEnv (ParserEnvWithGuards (withGuards SUD)) sqlSchema) (spiderSQL SUD mkParserStateUD)
         in case attoParseOnly p query of
              Left _ -> pure ()
              Right a -> H.assertFailure $ show a
    toTest (ParseQueryExprFailsTypeChecking sqlSchema query) =
      H.testCase ("Type checking fail " <> show query) $
        let p = withEnv (ParserEnv (ParserEnvWithGuards (withGuards STC)) sqlSchema) (spiderSQL STC mkParserStateTC)
         in case yoctoParseOnly p query of
              a : _ -> H.assertFailure $ show a
              [] -> pure ()
