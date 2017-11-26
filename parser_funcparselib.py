from funcparserlib.lexer import make_tokenizer, Token
from funcparserlib.parser import (some, maybe, many, skip, forward_decl)
import expr as e
from expr import EIdent, EQuantified, EApply, Thm, bin_ops

_tokenizer_specs = [
    ('Space', (r'[ \t\r\n]+',)),
    ('LParen', (r'\(',)),
    ('RParen', (r'\)',)),
    ('Turnstile', (r'\|-',)),
    ('Ident', (r"\.\.|,|@|#!#|#\?#|\?!|\?|!|lambda|\\/|/\\|\\|(?!\|-)[a-zA-Z0-9#^~$'_%+\-*<>=/|]+",)),
    ('Dot', (r'\.',)),
]
_tokenizer_useless = ['Space']
_tokenizer = make_tokenizer(_tokenizer_specs)

def tokenize(str):
    """str -> Sequence(Token)"""    
    return [x for x in _tokenizer(str) if x.type not in _tokenizer_useless]


_binder_names = frozenset(["@", "!", "?!", "?", "\\", "lambda"])

tokval = lambda t: t.value

ident = some(lambda t: t.type=='Ident')
binder = some(lambda t: t.type=='Ident' and t.value in _binder_names) >> tokval
binop = some(lambda t: t.type=='Ident' and t.value in bin_ops) >> tokval
lparen = skip(some(lambda t: t.type=='LParen'))
rparen = skip(some(lambda t: t.type=='RParen'))
dot = skip(some(lambda t: t.type=='Dot'))
comma = skip(some(lambda t: t.type=='Ident' and t.value== ","))

expr = forward_decl()    

expr_cont_quantified = (binder + ident + dot + expr + rparen) >> (lambda xs: EQuantified(*xs))

expr_cont_expr1 = (expr + (((binop + expr + rparen) >> (lambda xs: lambda e1: EApply(EApply(EIdent(xs[0]),e1),xs[1]))) | ((expr + rparen) >> (lambda e2: lambda e1: EApply(e1,e2))))) >> (lambda xs: xs[1](xs[0]))

expr.define((ident >> EIdent) | (lparen + (expr_cont_quantified | expr_cont_expr1)))

def sep_by(p1,p2):
    return maybe(p1 + many(p2 + p1)) >> (lambda xs: [] if xs is None else [xs[0]] + xs[1])

thm = (sep_by(expr, comma) + skip(some(lambda t: t.type=="Turnstile")) + expr) >> (lambda xs: Thm(xs[0], xs[1]))
