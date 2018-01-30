import pyparsing as pp
from pyparsing import Forward, Group, Literal, Optional, Or, Word, ZeroOrMore
import expr as e
from expr import EIdent, EQuantified, EApply, Thm

ident = Word("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@!^~?$'\_%+-*<>=/\\") | Literal("..") | Literal(",")
binder = Literal("@") | Literal("!") | Literal("?!") | Literal("?") | Literal("\\") | Literal("lambda")
lparen = Literal('(').suppress()
rparen = Literal(')').suppress()
dot = Literal(".").suppress()

expr = Forward()

expr_cont_quantified = (binder + ident + dot + expr + rparen).setParseAction(lambda toks: EQuantified(*toks))

binop = Or(Literal(s) for s in e.bin_ops)

expr_cont_expr1 = (expr + ((binop + expr + rparen).setParseAction(lambda toks: lambda e1: EApply(EApply(EIdent(toks[0]),e1),toks[1])) | (expr + rparen).setParseAction(lambda toks: lambda e1: EApply(e1,toks[0])))).setParseAction(lambda toks: toks[1](toks[0]))

expr << (ident.setParseAction(lambda toks: EIdent(*toks)) | (lparen + (expr_cont_quantified | expr_cont_expr1)))

def sep_by(p1,p2):
    p2 = p2.suppress()
    return Optional(p1 + ZeroOrMore(p2 + p1))

thm = (Group(sep_by(expr, Literal(",")).setParseAction(lambda toks: list(toks))) + Literal("|-").suppress() + expr).setParseAction(lambda toks: Thm(*toks))
