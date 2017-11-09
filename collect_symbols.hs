{-# OPTIONS_GHC -Wall #-}
import Control.Monad
import qualified Data.ByteString.Char8 as B
import Data.Foldable as F
import Data.IORef
import Data.Set (Set)
import qualified Data.Set as Set
import System.IO
import Text.Megaparsec
import Text.Printf
import Holstep

main :: IO ()
main = do
  ref <- newIORef Set.empty
  forM_ [(1::Int)..9999] $ \i -> do
    f ref (printf "holstep/train/%05d" i)
  forM_ [(1::Int)..1411] $ \i -> do
    f ref (printf "holstep/test/%04d" i)
  F.mapM_ B.putStrLn =<< readIORef ref

f :: IORef (Set B.ByteString) -> FilePath -> IO ()
f ref fname = do
  hPutStrLn stderr fname
  df <- readDataFile fname
  let ts = dfConjecture df : dfDependencies df ++ map fst (dfExamples df)
  forM_ ts $ \t -> do
    case parse (thm <* eof) fname (formulaText t) of
      Right (ps,c) -> modifyIORef' ref (`Set.union` Set.unions [fvars e | e <- c:ps])
      Left err -> do
        putStrLn $ B.unpack (formulaText t)
        putStrLn $ parseErrorPretty err
