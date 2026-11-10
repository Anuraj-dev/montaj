"""A tiny safe expression language for keyable layer properties (#52).

Parsed with `ast` and walked by a whitelist; never `eval`/`exec`. Allowed: numeric literals, `+ - * / % **`,
unary minus, comparisons (a true test is 1.0), `a if c else b`, and these names:

    t (seconds on the shot clock), frame (shot-clock frames), fps, beat (beats since the grid origin),
    bpm, word(N) (shot-clock seconds of audio word N's start), sin, cos, abs, min, max, clamp(x, lo, hi),
    lerp(a, b, x), noise(seed, x) (smooth value noise in [-1, 1]).

`bind` rewrites a validated expression so `fps`, `bpm`, `t`, `beat` and `word(N)` become constants and
`frame`; the renderer then needs only the frame, and the bound text is what the segment hash sees.
All arithmetic is float, so there are no big ints; a power is capped at |exponent| <= 64.
"""
from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from functools import lru_cache

MAX_LEN = 400
MAX_NODES = 120
MAX_POW = 64.0
NAMES = ("t", "frame", "fps", "beat", "bpm")
# name -> (min args, max args or None)
FUNCS: dict[str, tuple[int, int | None]] = {
    "sin": (1, 1), "cos": (1, 1), "abs": (1, 1), "min": (2, None), "max": (2, None),
    "clamp": (3, 3), "lerp": (3, 3), "noise": (2, 2), "word": (1, 1),
}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.Pow: "**"}
_CMPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
HINT = f"use numbers, + - * / % **, comparisons, a if c else b, {', '.join(NAMES)}, and " \
       f"{', '.join(n + '()' for n in FUNCS)}"


class ExprError(ValueError):
    """A bad expression. The message names the offending token."""


def _tok(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - a message helper must not fail
        return type(node).__name__


def _check(node: ast.AST, count: list[int]) -> None:
    count[0] += 1
    if count[0] > MAX_NODES:
        raise ExprError(f"too long: more than {MAX_NODES} terms")
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ExprError(f"{_tok(node)} is not a number")
        if not math.isfinite(float(v)) or abs(float(v)) > 1e9:
            raise ExprError(f"number {_tok(node)} is out of range (|x| <= 1e9)")
    elif isinstance(node, ast.Name):
        if node.id not in NAMES:
            raise ExprError(f"unknown name {node.id!r}")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ExprError(f"operator {_tok(node)!r} is not allowed")
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and abs(float(node.right.value)) > MAX_POW:
            raise ExprError(f"exponent {_tok(node.right)} is too large (|exponent| <= {int(MAX_POW)})")
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ExprError(f"operator in {_tok(node)!r} is not allowed")
    elif isinstance(node, ast.Compare):
        if any(type(op) not in _CMPS for op in node.ops):
            raise ExprError(f"comparison {_tok(node)!r} is not allowed")
    elif isinstance(node, ast.IfExp):
        pass
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError(f"call {_tok(node.func)!r} is not allowed")
        name = node.func.id
        if name not in FUNCS:
            raise ExprError(f"unknown function {name!r}")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise ExprError(f"{name}() takes plain arguments")
        lo, hi = FUNCS[name]
        if len(node.args) < lo or (hi is not None and len(node.args) > hi):
            want = str(lo) if lo == hi else f"{lo} or more" if hi is None else f"{lo} to {hi}"
            raise ExprError(f"{name}() takes {want} arguments, got {len(node.args)}")
        if name == "word":
            a = node.args[0]
            if not (isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool)
                    and a.value >= 0):
                raise ExprError(f"word({_tok(a)}) needs a whole number like word(3)")
        for arg in node.args:
            _check(arg, count)
        return
    elif isinstance(node, ast.Attribute):
        raise ExprError(f"attribute access {_tok(node)!r} is not allowed")
    elif isinstance(node, ast.Subscript):
        raise ExprError(f"indexing {_tok(node)!r} is not allowed")
    elif isinstance(node, (ast.Load, ast.operator, ast.unaryop, ast.cmpop)):
        return
    else:
        raise ExprError(f"{type(node).__name__} {_tok(node)!r} is not allowed")
    for child in ast.iter_child_nodes(node):
        _check(child, count)


@lru_cache(maxsize=256)
def parse(text: str) -> ast.Expression:
    """The validated tree for `text`, or `ExprError` naming the bad token. Memoised on the text (pure)."""
    if not isinstance(text, str) or not text.strip():
        raise ExprError("empty expression")
    if len(text) > MAX_LEN:
        raise ExprError(f"too long: {len(text)} characters (max {MAX_LEN})")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        bad = (e.text or text).strip()
        near = bad[max(0, (e.offset or 1) - 1):][:12] or bad[-12:]
        raise ExprError(f"syntax error near {near!r}") from None
    except (ValueError, RecursionError):
        raise ExprError("cannot parse") from None
    _check(tree.body, [0])
    return tree


def validate(text: str) -> None:
    parse(text)


def words(text: str) -> list[int]:
    """The N of every `word(N)` call."""
    return sorted({n.args[0].value for n in ast.walk(parse(text))
                   if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "word"})


def uses(text: str, *names: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(parse(text)))


class _Bind(ast.NodeTransformer):
    def __init__(self, fps: float, bpm: float | None, start: float, beat0: float, word_frames: Mapping[int, float]):
        self.fps, self.bpm, self.start, self.beat0, self.words = fps, bpm, start, beat0, word_frames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        frame = ast.Name(id="frame", ctx=ast.Load())
        if node.id == "fps":
            return ast.Constant(self.fps)
        if node.id == "bpm":
            if not self.bpm:
                raise ExprError("bpm needs video.bpm")
            return ast.Constant(float(self.bpm))
        if node.id == "t":
            return ast.BinOp(frame, ast.Div(), ast.Constant(float(self.fps)))
        if node.id == "beat":
            if not self.bpm:
                raise ExprError("beat needs video.bpm")
            bf = self.fps * 60.0 / self.bpm  # frames per beat
            shifted = ast.BinOp(frame, ast.Add(), ast.Constant(float(self.start - self.beat0)))
            return ast.BinOp(shifted, ast.Div(), ast.Constant(bf))
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if node.func.id == "word":
            n = node.args[0].value
            if n not in self.words:
                raise ExprError(f"unknown word {n}")
            return ast.Constant((self.words[n] - self.start) / self.fps)
        node.args = [self.visit(a) for a in node.args]
        return node


def bind(text: str, *, fps: float, bpm: float | None, start: float, beat0: float,
         word_frames: Mapping[int, float]) -> str:
    """`text` with the film's constants folded in: only `frame` and the functions remain. `start` is the shot's
    first film frame, `beat0` the beat grid origin in film frames, `word_frames` word N's start in film frames."""
    parse(text)  # validate; the cached tree is shared, so transform a fresh one
    tree = _Bind(fps, bpm, start, beat0, word_frames).visit(ast.parse(text.strip(), mode="eval"))
    return ast.unparse(ast.fix_missing_locations(tree).body)


def _noise_at(seed: float, i: int) -> float:
    n = (int(seed) * 374761393 + i * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    n ^= n >> 16
    return n / 0xFFFFFFFF * 2.0 - 1.0


def _noise(seed: float, x: float) -> float:
    i = math.floor(x)
    f = x - i
    u = f * f * (3.0 - 2.0 * f)
    a = _noise_at(seed, i)
    return a + (_noise_at(seed, i + 1) - a) * u


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


_FN: dict[str, Callable[..., float]] = {
    "sin": math.sin, "cos": math.cos, "abs": abs, "min": min, "max": max, "clamp": _clamp,
    "lerp": lambda a, b, x: a + (b - a) * x, "noise": _noise,
}


def _pow(a: float, b: float) -> float:
    if abs(b) > MAX_POW:
        raise ExprError(f"exponent {b:g} is too large (|exponent| <= {int(MAX_POW)})")
    return a**b


def _eval(node: ast.AST, env: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExprError(f"{node.id} is not available here")
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.Mod):
            return a % b
        return _pow(a, b)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            ok = {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right,
                  ast.GtE: left >= right, ast.Eq: left == right, ast.NotEq: left != right}[type(op)]
            if not ok:
                return 0.0
            left = right
        return 1.0
    if isinstance(node, ast.IfExp):
        return _eval(node.body if _eval(node.test, env) != 0.0 else node.orelse, env)
    if isinstance(node, ast.Call):
        if node.func.id == "word":
            raise ExprError("word() is only available after the film is resolved")
        return float(_FN[node.func.id](*(_eval(a, env) for a in node.args)))
    raise ExprError(f"{type(node).__name__} is not allowed")


def evaluate(text: str, env: Mapping[str, float]) -> float:
    """The value of `text` for `env` (e.g. `{"frame": 12.0}` for bound text); finite or `ExprError`."""
    try:
        v = _eval(parse(text).body, env)
    except ExprError:
        raise
    except (ArithmeticError, ValueError) as e:
        raise ExprError(f"{e} at {dict(env)}") from None
    if not math.isfinite(v):
        raise ExprError(f"not finite at {dict(env)}")
    return v
