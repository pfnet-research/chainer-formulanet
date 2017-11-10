import parsy
from parsy import alt, fail, generate, regex, seq, string

class Expr(object):
    pass

class EIdent(Expr):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class EApply(Expr):
    def __init__(self, fun, arg):
        self.fun = fun
        self.arg = arg

    # 注意: 中置演算子なども前置の正しくない形式で文字列化される
    def __str__(self):
        return "({} {})".format(str(self.fun), str(self.arg))

class EQuantified(Expr):
    def __init__(self, binder, var, body):
        self.binder = binder
        self.var = var
        self.body = body

    def __str__(self):
        return "({} {}. {})".format(self.binder, self.var, str(self.body))

class Thm:
    def __init__(self, premises, conclusion):
        self.premises = tuple(premises)
        self.conclusion = conclusion

    def __str__(self):
        return "{} |- {}".format(",".join(map(str, self.premises)), str(self.conclusion))


spaces = regex(r'\s*')

def lexme(p):
    return (p << spaces)

ident = lexme(regex(r"([a-zA-Z0-9#@!^~?$'\_%+\-*<>=/\\])+") | string("..") | string(",")).desc("identifier")

lparen = lexme(string('('))

rparen = lexme(string(')'))

dot = lexme(string("."))

bin_ops = frozenset(["<=>", "==>", "\\/", "/\\", "==", "===", "treal_eq", "IN", "<", "<<", "<<<", "<<=", "<=", "<=_c", "<_c", "=", "=_c", ">", ">=", ">=_c", ">_c", "HAS_SIZE", "PSUBSET", "SUBSET", "divides", "has_inf", "has_sup", "treal_le", ",", "..", "+", "++", "UNION", "treal_add", "-", "DIFF", "*", "**", "INTER", "INTERSECTION_OF", "UNION_OF", "treal_mul", "INSERT", "DELETE", "CROSS", "PCROSS", "/", "DIV", "MOD", "div", "rem", "EXP", "pow", "$", "o", "%", "%%", "-->", "--->"])

@generate
def expr_cont_quantified():
    b = yield lexme(regex(r"@|!|\?!|\?|\\|lambda"))
    v = yield ident
    yield dot
    e = yield expr
    yield rparen
    return EQuantified(b,v,e)

@generate
def binop():
    op = yield ident
    if op not in bin_ops:
        yield fail("{} is not a binary operator".format(op))
    else:
        return op

@generate
def expr_cont_expr1():
    e1 = yield expr
    p1 = seq(binop, expr << rparen).combine(lambda op,e2: EApply(EApply(EIdent(op),e1),e2))
    p2 = (expr << rparen).map(lambda e2: EApply(e1,e2))
    return (yield (p1 | p2))

expr = alt(ident.map(EIdent), (lparen >> (expr_cont_quantified | expr_cont_expr1)))

thm = seq(expr.sep_by(lexme(string(","))), lexme(string("|-")) >> expr).combine(Thm)
