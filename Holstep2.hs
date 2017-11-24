{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE OverloadedStrings #-}
module Holstep2
  ( Formula (..)
  , DataFile (..)
  , readDataFile

  , Expr (..)
  , Thm
  , expr
  , thm

  , fvars
  ) where

import Control.Applicative
import Control.Monad.State.Strict
import Data.Attoparsec.ByteString.Char8
import qualified Data.Attoparsec.ByteString.Lazy as AttoparsecLazy
import qualified Data.ByteString.Char8 as B
import qualified Data.ByteString.Lazy as BL
import Data.List
import Data.Ord
import Data.Set (Set)
import qualified Data.Set as Set

data Formula
  = Formula
  { formulaName   :: Maybe B.ByteString
  , formulaText   :: !B.ByteString
  , formulaTokens :: !B.ByteString
  }
  deriving (Show)

data DataFile
  = DataFile
  { dfConjecture   :: !Formula
  , dfDependencies :: [Formula]
  , dfExamples     :: [(Formula, Bool)]
  }
  deriving (Show)

readDataFile :: FilePath -> IO DataFile
readDataFile name = do
  s <- BL.readFile name
  case AttoparsecLazy.eitherResult (AttoparsecLazy.parse dataFileParser s) of
    Left e -> error (show e)
    Right x -> return x

dataFileParser :: Parser DataFile
dataFileParser = DataFile <$> conjParser <*> many depParser <*> many statementParser

conjParser :: Parser Formula
conjParser = Formula <$> (Just <$> prefixedLine 'N') <*> prefixedLine 'C' <*> prefixedLine 'T'

depParser :: Parser Formula
depParser = Formula <$> (Just <$> prefixedLine 'D') <*> prefixedLine 'A' <*> prefixedLine 'T'

statementParser :: Parser (Formula, Bool)
statementParser = msum
  [ do s <- prefixedLine '+'
       t <- prefixedLine 'T'
       return (Formula Nothing s t, True)
  , do s <- prefixedLine '-'
       t <- prefixedLine 'T'
       return (Formula Nothing s t, False)
  ]

prefixedLine :: Char -> Parser B.ByteString
prefixedLine c = char c *> space *> (B.pack <$> manyTill anyChar (try endOfLine))


type Ident = B.ByteString

type Binder = Ident

data Expr
  = EIdent Ident
  | EApply Expr Expr
  | EQuantified Binder Ident Expr
  deriving (Show)

type Thm = ([Expr], Expr)

lexme :: Parser a -> Parser a
lexme p = p <* many space

ident :: Parser Ident
ident = lexme $ try (p <?> "ident")
  where
    p = fmap B.pack (many1 (satisfy (\c -> isAlpha_ascii c || isDigit c || c `elem` ['#','@','!','^','~','?','$','\'','_','%','+','-','*','<','>','=','/','\\'])))
     <|> string ".."
     <|> string ","

lparen :: Parser ()
lparen = lexme $ char '(' *> pure ()

rparen :: Parser ()
rparen = lexme $ char ')' *> pure ()

dot :: Parser ()
dot = lexme $ char '.' *> pure ()

expr :: Parser Expr
expr = msum
  [ EIdent <$> ident
  , do lparen
       msum
         [ do (b,v) <- try $ do
                b <- lexme $ msum [string b | b <- ["@", "!", "?!", "?", "\\", "lambda"]]
                v <- ident
                dot
                return (b,v)
              e <- expr
              rparen
              return $ EQuantified b v e
         , do e1 <- expr
              msum
                [ try $ do
                    op <- ident
                    guard $ op `Set.member` binOps_set
                    e2 <- expr
                    rparen
                    return $ EApply (EApply (EIdent op) e1) e2
                , do e2 <- expr
                     rparen
                     return $ EApply e1 e2
                ]
         ]
  ]

thm :: Parser Thm
thm = (,)
  <$> (expr `sepBy` lexme (char ','))
  <*> (lexme (string "|-") *> expr)

binOps_set :: Set B.ByteString
binOps_set = Set.fromList binOps

binOps_sorted :: [B.ByteString]
binOps_sorted = sortBy (comparing (negate . B.length)) binOps

binOps :: [B.ByteString]
binOps =
  [ "<=>"
  , "==>"
  , "\\/"
  , "/\\"
  , "=="
  , "==="
  , "treal_eq"
  , "IN"
  , "<"
  , "<<"
  , "<<<"
  , "<<="
  , "<="
  , "<=_c"
  , "<_c"
  , "="
  , "=_c"
  , ">"
  , ">="
  , ">=_c"
  , ">_c"
  , "HAS_SIZE"
  , "PSUBSET"
  , "SUBSET"
  , "divides"
  , "has_inf"
  , "has_sup"
  , "treal_le"
  , ","
  , ".."
  , "+"
  , "++"
  , "UNION"
  , "treal_add"
  , "-"
  , "DIFF"
  , "*"
  , "**"
  , "INTER"
  , "INTERSECTION_OF"
  , "UNION_OF"
  , "treal_mul"
  , "INSERT"
  , "DELETE"
  , "CROSS"
  , "PCROSS"
  , "/"
  , "DIV"
  , "MOD"
  , "div"
  , "rem"
  , "EXP"
  , "pow"
  , "$"
  , "o"
  , "%"
  , "%%"
  , "-->"
  , "--->"
  ]


fvars :: Expr -> Set Ident
fvars (EIdent x) = Set.singleton x
fvars (EApply e1 e2) = fvars e1 `Set.union` fvars e2
fvars (EQuantified _b v e) = Set.delete v (fvars e)
