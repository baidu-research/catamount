import sympy


def getIntSymbolFromString(sym_name):
    assert isinstance(sym_name, str)
    # Integer symbols should specify explicitly (e.g., Dimensions)
    return sympy.Symbol(sym_name, integer=True)

def getPositiveIntSymbolFromString(sym_name):
    assert isinstance(sym_name, str)
    # Integer symbols should specify explicitly (e.g., Dimensions)
    return sympy.Symbol(sym_name, integer=True, positive=True)

def getSymbolicMaximum(expr_0, expr_1, symbol_subs=None):
    if symbol_subs is not None:
        if isinstance(expr_0, sympy.Expr):
            expr_0 = expr_0.subs(symbol_subs)
        if isinstance(expr_1, sympy.Expr):
            expr_1 = expr_1.subs(symbol_subs)
    # Try to clamp expressions to ints
    try:
        expr_0 = int(expr_0)
    except:
        pass
    try:
        expr_1 = int(expr_1)
    except:
        pass
    if isinstance(expr_0, sympy.Expr) or isinstance(expr_1, sympy.Expr):
        print('WARN: Symbolic maximum is slow! {} {}'.format(expr_0, expr_1))
        return sympy.functions.Max(expr_0, expr_1)
    else:
        return max(expr_0, expr_1)
