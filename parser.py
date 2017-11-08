import parsy
from parsy import alt, generate, regex, seq, string

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

class ETuple(Expr):
    def __init__(self, args):
        self.args = args

    def __str__(self):
        return "(" + ", ".join(map(str, self.args)) + ")"

class EQuantified(Expr):
    def __init__(self, binder, var, body):
        self.binder = binder
        self.var = var
        self.body = body

    def __str__(self):
        return "({} {}. {})".format(self.binder, self.var, str(self.body))

class Judgement:
    def __init__(self, premises, conclusion):
        self.premises = tuple(premises)
        self.conclusion = conclusion

    def __str__(self):
        return "{} |- {}".format(",".join(map(str, self.premises)), str(self.conclusion))


spaces = regex(r'\s*')

def lexme(p):
    return (p << spaces)

def paren(p):
    return (lexme(string('(')) >> p << lexme(string(')')))

comma = lexme(string(","))
dot = lexme(string("."))
turnstile = lexme(string("|-"))
ident = lexme(regex("[a-zA-Z_](\w|[_%'])*") | string("?")).desc("identifier")
binop = lexme(alt(*[string(s) for s in ["$", "..", "-->", "--->", "==>", "==", "=_c", "=", "<=_c", "<=", ">=", ">=_c", "<_c", "<", ">_c", ">", "/\\", "\\/", "++", "+_c", "+", "-_c", "-", "*_c", "*", "^_c", "^", "%%", "%", "IN", "INSERT", "o"]])).desc("binary operator")
uop = lexme(string("~") | string("@")).desc("unary operator")
binder = lexme(alt(*[string(s) for s in ["@", "!", "?!", "?", "\\", "lambda"]])).desc("binder")

@generate
def expr_atom():
    binop_pap = seq(binop.map(EIdent), expr_uop).combine(EApply)
    tuple = seq(expr, (comma >> expr).many()).combine(lambda e, es: e if len(es)==0 else ETuple([e] + es))
    unknown = seq(expr_uop, (binop | lexme(string(","))).map(EIdent)).combine(lambda e, op: EApply(e, op)) # XXX: 解釈が正しいか不明
    return (yield (ident.map(EIdent) | paren(binop_pap | tuple) | paren(unknown))) # paren(binop_pap | tuple | unknown) とはできないので注意

@generate
def expr_apply():
    e = yield expr_atom
    args = yield expr_atom.many()
    for arg in args:
        e = EApply(e, arg)
    return e

expr_uop = seq(uop.map(EIdent), expr_apply).combine(lambda op, e: EApply(op, e)) | expr_apply

@generate
def expr_binop():
    e = yield expr_uop
    rest = yield seq(binop.map(EIdent), expr_uop).many()
    for op, e2 in rest:
        e = EApply(EApply(op, e), e2)
    return e

@generate
def expr_quantified():
    b = yield binder
    v = yield ident
    yield dot
    body = yield expr
    return EQuantified(b,v,body)

expr = (expr_quantified | expr_binop)

judgement = seq(expr.sep_by(comma), turnstile >> expr).combine(Judgement)
