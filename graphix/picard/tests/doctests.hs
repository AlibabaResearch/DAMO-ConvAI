module Main where

import Build_doctests (Component (..), components)
import Data.Foldable (for_)
import System.Environment.Compat (unsetEnv)
import Test.DocTest (doctest)

main :: IO ()
main = for_ components $ \(Component name flags pkgs sources) -> do
  print name
  putStrLn "----------------------------------------"
  let args = flags ++ pkgs ++ sources
  for_ args putStrLn
  unsetEnv "GHC_ENVIRONMENT"
  doctest args
