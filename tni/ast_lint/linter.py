"""A best-effort, EXPLICITLY UNSOUND AST linter for look-ahead bias in ordinary
pandas/numpy/sklearn code (DESIGN.md B.8).

Unlike the tni core -- which is sound because it propagates availability labels
-- this linter just pattern-matches syntactic red flags in source text. It has
both false negatives (it cannot see leaks that aren't a syntactic signature) and
false positives (it cannot prove a flagged construct is actually misused). It is
a bug-finder for legacy code, NOT a verifier, and it does not touch the
soundness theorem (a statement about the core only).

Detected signatures:
  * negative .shift(-k)            -> forward shift / lead
  * .rolling(center=True)          -> centered window peeks ahead
  * KFold(shuffle=True) / train_test_split without shuffle=False
                                   -> shuffled CV on (likely) time series
  * scaler .fit_transform(X)       -> normalisation fit on the whole sample
  * (x - x.mean()) / x.std() etc.  -> full-sample standardisation
"""
import ast
from dataclasses import dataclass

_REDUCERS = {"mean", "std", "var", "min", "max", "median", "quantile", "sum"}
_WINDOWERS = {"rolling", "expanding", "ewm"}
_SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
            "PowerTransformer", "QuantileTransformer"}


@dataclass
class Finding:
    line: int
    code: str
    message: str


def _is_negative(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and \
            isinstance(node.operand.value, (int, float)) and node.operand.value > 0
    return isinstance(node, ast.Constant) and \
        isinstance(node.value, (int, float)) and node.value < 0


def _is_scaler_ctor(node):
    """True if node is `SomeScaler(...)` for a known scaler class."""
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id in _SCALERS) or
        (isinstance(node.func, ast.Attribute) and node.func.attr in _SCALERS))


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_true(node):
    return isinstance(node, ast.Constant) and node.value is True


def _windowed_receiver(call):
    """True if a reducer call's receiver chain includes rolling/expanding/ewm."""
    recv = call.func.value if isinstance(call.func, ast.Attribute) else None
    while recv is not None:
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) \
                and recv.func.attr in _WINDOWERS:
            return True
        recv = recv.func.value if isinstance(recv, ast.Call) and \
            isinstance(recv.func, ast.Attribute) else None
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.findings = []

    def _add(self, node, code, msg):
        self.findings.append(Finding(getattr(node, "lineno", 0), code, msg))

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute):
            # negative shift -> lead
            if f.attr == "shift" and node.args and _is_negative(node.args[0]):
                self._add(node, "TNI-LEAD", "negative .shift() reads future rows")
            # centered rolling
            if f.attr == "rolling" and _is_true(_kw(node, "center")):
                self._add(node, "TNI-CENTER", ".rolling(center=True) peeks ahead")
            # scaler fit on the whole sample: fit_transform always, or a bare
            # ScalerClass().fit(...) (but not a pipelined model.fit())
            if f.attr == "fit_transform" or (
                    f.attr == "fit" and _is_scaler_ctor(f.value)):
                self._add(node, "TNI-FITALL",
                          f".{f.attr}() likely fits a scaler on the whole "
                          "sample (normalise after the train split)")
            # full-sample reduction used outside a window
            if f.attr in _REDUCERS and not _windowed_receiver(node):
                # only flag when it feeds arithmetic is handled in visit_BinOp;
                # here flag bare full-sample .quantile()/.std() style stats
                pass
        if isinstance(f, ast.Name):
            if f.id == "KFold" and _is_true(_kw(node, "shuffle")):
                self._add(node, "TNI-CV", "KFold(shuffle=True) on time series leaks")
            if f.id == "train_test_split":
                sh = _kw(node, "shuffle")
                if sh is None or _is_true(sh):
                    self._add(node, "TNI-CV",
                              "train_test_split shuffles by default (pass "
                              "shuffle=False for time series)")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # (x - x.mean()) or (... / x.std()) with a non-windowed reducer
        for operand in (node.left, node.right):
            if isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute) \
                    and operand.func.attr in _REDUCERS and not _windowed_receiver(operand):
                self._add(node, "TNI-FULLSTAT",
                          f"full-sample .{operand.func.attr}() in arithmetic "
                          "(use a trailing window)")
        self.generic_visit(node)


def lint_source(src, filename="<string>"):
    try:
        tree = ast.parse(src, filename=filename)
    except SyntaxError as e:
        return [Finding(e.lineno or 0, "TNI-PARSE", f"could not parse: {e.msg}")]
    v = _Visitor()
    v.visit(tree)
    return sorted(v.findings, key=lambda f: (f.line, f.code))


def lint_file(path):
    with open(path) as f:
        return lint_source(f.read(), filename=path)
