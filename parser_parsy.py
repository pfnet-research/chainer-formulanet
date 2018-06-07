from parsy import alt, fail, generate, regex, seq, string
import expr as e
from expr import EIdent, EQuantified, EApply, Thm

spaces = regex(r'\s*')


def lexme(p):
    return p << spaces


ident = lexme(regex(r"([a-zA-Z0-9#@!^~?$'\_%+\-*<>=/\\])+") | string("..") | string(",")).desc("identifier")

lparen = lexme(string('('))

rparen = lexme(string(')'))

dot = lexme(string("."))


@generate
def expr_cont_quantified():
    b = yield lexme(regex(r"@|!|\?!|\?|\\|lambda"))
    v = yield ident
    yield dot
    body = yield expr
    yield rparen
    return EQuantified(b, v, body)


@generate
def binop():
    op = yield ident
    if op not in e.bin_ops:
        yield fail("{} is not a binary operator".format(op))
    else:
        return op


@generate
def expr_cont_expr1():
    e1 = yield expr
    p1 = seq(binop, expr << rparen).combine(lambda op, e2: EApply(EApply(EIdent(op), e1), e2))
    p2 = (expr << rparen).map(lambda e2: EApply(e1, e2))
    return (yield (p1 | p2))


expr = alt(ident.map(EIdent), (lparen >> (expr_cont_quantified | expr_cont_expr1)))

thm = seq(expr.sep_by(lexme(string(","))), lexme(string("|-")) >> expr).combine(Thm)
