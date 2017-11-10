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

bin_ops = frozenset(["<=>", "==>", "\\/", "/\\", "==", "===", "treal_eq", "IN", "<", "<<", "<<<", "<<=", "<=", "<=_c", "<_c", "=", "=_c", ">", ">=", ">=_c", ">_c", "HAS_SIZE", "PSUBSET", "SUBSET", "divides", "has_inf", "has_sup", "treal_le", ",", "..", "+", "++", "UNION", "treal_add", "-", "DIFF", "*", "**", "INTER", "INTERSECTION_OF", "UNION_OF", "treal_mul", "INSERT", "DELETE", "CROSS", "PCROSS", "/", "DIV", "MOD", "div", "rem", "EXP", "pow", "$", "o", "%", "%%", "-->", "--->"])
