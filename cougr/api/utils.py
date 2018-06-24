import sympy


def getIntSymbolFromString(sym_name):
    assert isinstance(sym_name, str)
    # Integer symbols should specify explicitly (e.g., Dimensions)
    return sympy.Symbol(sym_name, integer=True)

