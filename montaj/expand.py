"""Spec reuse: `include`, `styles`, `defaults`, `components`, `use` -> a plain spec (#42).

A pure YAML->YAML step that runs before validation. The expanded spec is what gets validated and
hashed, so reuse never changes cache behaviour. `SourceMap` maps expanded paths back to the author's
key (and line), so errors point at what the author wrote.

Precedence for a text line: its own fields > its `styles:` entry > `defaults`.
Dicts merge recursively; an explicit `null` wins (`in: null` drops a default transition).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REUSE_KEYS = ("include", "styles", "defaults", "components", "use")
_LISTS = ("shots", "text", "fx", "subs", "overlays")
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_ARITH = re.compile(r"^\$([A-Za-z_]\w*)\s*([+-])\s*(\d+(?:\.\d+)?)([sfb])$")
_PARAM = re.compile(r"\$([A-Za-z_]\w*)")
_DUR = re.compile(r"^(\d+(?:\.\d+)?)([sfb])$")
_OFFSET = re.compile(r"^(.*?)([+-])(\d+(?:\.\d+)?)([sfb])$")
_REQUIRED = object()


class ExpandError(Exception):
    """Authoring problems: (author path, problem, hint) triples."""

    def __init__(self, errors: list[tuple[str, str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{p}: {m} ({h})" for p, m, h in errors))

    def lines(self, text: str) -> list[str]:
        out = []
        for path, problem, hint in self.errors:
            line = _line_of(text, path)
            where = f"{path} (line {line})" if line is not None else path
            out.append(f"{where}: {problem} ({hint})")
        return out


@dataclass
class _Source:
    name: str  # "" for the spec itself, else the include path as written
    text: str


@dataclass
class SourceMap:
    sources: list[_Source] = field(default_factory=list)
    # expanded path -> (source index, author path). Longest prefix wins.
    entries: dict[str, tuple[int, str]] = field(default_factory=dict)

    def author(self, path: str) -> tuple[int, str]:
        for cut in _prefixes(path):
            if cut in self.entries:
                src, mapped = self.entries[cut]
                return src, mapped + path[len(cut):]
        return 0, path

    def remap(self, message: str) -> str:
        """Rewrite the leading path of an error line to the author's key and line."""
        path, sep, rest = message.partition(": ")
        if not sep:
            return message
        src, mapped = self.author(path)
        line = _line_of(self.sources[src].text, mapped)
        name = self.sources[src].name
        label = f"{name}:{mapped}" if name else mapped
        if line is not None:
            label += f" (line {line})"
        return f"{label}: {rest}"


def _prefixes(path: str):
    """The path, then each shorter prefix cut at a `.` or `[` boundary."""
    yield path
    for i in range(len(path) - 1, 0, -1):
        if path[i] in ".[":
            yield path[:i]


def _line_of(text: str, path: str) -> int | None:
    try:
        node = yaml.compose(text)
    except yaml.YAMLError:
        return None
    for m in _TOKEN.finditer(path):
        key, idx = m.groups()
        if node is None:
            return None
        if idx is not None:
            if not isinstance(node, yaml.SequenceNode) or int(idx) >= len(node.value):
                return None
            node = node.value[int(idx)]
        else:
            if not isinstance(node, yaml.MappingNode):
                return None
            for k, v in node.value:
                if k.value == key:
                    node = v
                    break
            else:
                return None
    return None if node is None else node.start_mark.line + 1


def _err(path: str, problem: str, hint: str) -> ExpandError:
    return ExpandError([(path, problem, hint)])


class _Expander:
    def __init__(self, spec_path: Path, builtin_styles: tuple[str, ...]):
        self.builtin = tuple(builtin_styles)
        self.smap = SourceMap()
        self.root = spec_path
        # name -> (source index, author path, body)
        self.styles: dict[str, tuple[int, str, dict]] = {}
        self.defaults: dict[str, tuple[int, str, dict]] = {}
        self.components: dict[str, tuple[int, str, dict]] = {}

    # --- loading -------------------------------------------------------------------------

    def load(self, data: dict, text: str) -> None:
        self.smap.sources.append(_Source("", text))
        self._collect(data, 0, self.root, [self.root.resolve()])

    def _collect(self, data: dict, src: int, file: Path, chain: list[Path]) -> None:
        inc = data.get("include")
        if inc is not None:
            names = inc if isinstance(inc, list) else [inc]
            if not all(isinstance(n, str) for n in names):
                raise _err("include", "need a path or list of paths", "include: ../brand.yaml")
            for i, name in enumerate(names):
                self._include(name, file, chain, f"include[{i}]" if isinstance(inc, list) else "include")
        for key, bucket in (("styles", self.styles), ("defaults", self.defaults), ("components", self.components)):
            block = data.get(key)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise _err(key, "need a mapping", f"{key}: {{name: ...}}")
            for name, body in block.items():
                if not isinstance(body, dict):
                    raise _err(f"{key}.{name}", "need a mapping", "give it fields")
                bucket[str(name)] = (src, f"{key}.{name}", body)  # the includer overrides what it includes

    def _include(self, name: str, file: Path, chain: list[Path], path: str) -> None:
        target = (file.parent / name).resolve()
        if target in chain:
            raise _err(path, f"include cycle through {name}", "remove the circular include")
        try:
            text = target.read_text()
        except OSError as e:
            raise _err(path, f"cannot read {name}", str(e)) from e
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise _err(path, f"invalid YAML in {name}", str(e)) from e
        if not isinstance(data, dict):
            raise _err(path, f"{name} must be a mapping", "styles:, defaults:, components:")
        extra = sorted(set(data) - {"include", "styles", "defaults", "components"})
        if extra:
            raise _err(path, f"{name} may only hold styles, defaults, components; has {', '.join(extra)}", "move them to the spec")
        self.smap.sources.append(_Source(name, text))
        self._collect(data, len(self.smap.sources) - 1, target, [*chain, target])

    # --- styles --------------------------------------------------------------------------

    def check_style_names(self) -> None:
        for name, (_, path, _) in self.styles.items():
            if name in self.builtin:
                raise _err(path, f"{name!r} is a built-in text style", f"pick another name; built-ins: {', '.join(self.builtin)}")
        for name in self.styles:
            self.flat_style(name)  # cycles and bad extends surface even when unused

    def flat_style(self, name: str) -> tuple[dict, dict[str, tuple[int, str]]]:
        """A style with `extends` resolved: (fields, field -> (source, author path))."""
        chain: list[str] = []
        cur = name
        while True:
            if cur in chain:
                raise _err(self.styles[chain[0]][1], f"extends cycle {' -> '.join([*chain, cur])}", "break the loop")
            chain.append(cur)
            parent = self.styles[cur][2].get("extends")
            if parent is None:
                break
            if parent not in self.styles:
                have = ", ".join(sorted(self.styles)) or "none"
                raise _err(f"{self.styles[cur][1]}.extends", f"unknown style {parent!r}", f"have: {have}")
            cur = parent
        out: dict = {}
        origin: dict[str, tuple[int, str]] = {}
        for link in reversed(chain):  # base first, children override
            src, path, body = self.styles[link]
            for k, v in body.items():
                if k == "extends":
                    continue
                out[k] = _deep_override(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
                origin[k] = (src, f"{path}.{k}")
        sty = out.get("style")
        if sty is not None and sty not in self.builtin:
            raise _err(f"{self.styles[name][1]}.style", f"{sty!r} is not a built-in text style", f"use {', '.join(self.builtin)}; reuse another style with extends")
        return out, origin

    # --- components ----------------------------------------------------------------------

    def use(self, data: dict) -> None:
        """Splice `{use: name, ...}` items in place; append top-level `use:` instances to their lists."""
        for lst in _LISTS:
            items = data.get(lst)
            if not isinstance(items, list) or not any(isinstance(i, dict) and "use" in i for i in items):
                continue
            out: list = []
            for i, item in enumerate(items):
                if isinstance(item, dict) and "use" in item:
                    call = {("component" if k == "use" else k): v for k, v in item.items()}
                    made = self._instantiate(call, f"{lst}[{i}]", data)
                    here = made.pop(lst, [])
                    for j in range(len(here)):
                        self.smap.entries[f"{lst}[{len(out) + j}]"] = (0, f"{lst}[{i}]")
                    out.extend(here)
                    for other, extra in made.items():
                        data.setdefault(other, []).extend(extra)
                else:
                    self.smap.entries[f"{lst}[{len(out)}]"] = (0, f"{lst}[{i}]")
                    out.append(item)
            data[lst] = out
        uses = data.get("use")
        if uses is None:
            return
        if not isinstance(uses, list):
            raise _err("use", "need a list", "use: [{component: name, ...}]")
        for k, call in enumerate(uses):
            for lst, made in self._instantiate(call, f"use[{k}]", data).items():
                target = data.setdefault(lst, [])
                for item in made:
                    self.smap.entries[f"{lst}[{len(target)}]"] = (0, f"use[{k}]")
                    target.append(item)

    def _instantiate(self, call: Any, where: str, data: dict) -> dict[str, list]:
        """One component call -> {list name: new items}."""
        if not isinstance(call, dict) or "component" not in call:
            raise _err(where, "need a mapping with component:", "{component: name, param: value}")
        name = call["component"]
        if name not in self.components:
            have = ", ".join(sorted(self.components)) or "none"
            raise _err(f"{where}.component", f"unknown component {name!r}", f"have: {have}")
        _, cpath, body = self.components[name]
        params = self._params(body, cpath)
        args = {a: v for a, v in call.items() if a != "component"}
        for a in args:
            if a not in params:
                raise _err(f"{where}.{a}", f"{name} has no param {a!r}", f"params: {', '.join(params) or 'none'}")
        vals: dict[str, Any] = {}
        for p, dflt in params.items():
            if p in args:
                vals[p] = args[p]
            elif dflt is not _REQUIRED:
                vals[p] = dflt
            else:
                raise _err(where, f"{name} needs param {p!r}", f"add {p}: ...")
        out: dict[str, list] = {}
        for lst, items in body.items():
            if lst == "params":
                continue
            if lst not in _LISTS or not isinstance(items, list):
                raise _err(f"{cpath}.{lst}", "components hold lists named " + ", ".join(_LISTS), "plus params")
            out[lst] = [_substitute(item, vals, f"{cpath}.{lst}[{j}]") for j, item in enumerate(items)]
        return out

    @staticmethod
    def _params(body: dict, cpath: str) -> dict[str, Any]:
        raw = body.get("params", [])
        if isinstance(raw, list):
            return {str(p): _REQUIRED for p in raw}
        if isinstance(raw, dict):
            return {str(p): v for p, v in raw.items()}
        raise _err(f"{cpath}.params", "need a list or mapping", "params: [word, at] or {at: 0s}")

    # --- styles and defaults on the tree -------------------------------------------------

    def apply(self, data: dict) -> None:
        lines_default = self.defaults.get("text.lines")
        styled: set[str] = set()
        for path, line in _targets(data, "text.lines"):
            name = line.get("style")
            if name is None and lines_default is not None:
                name = lines_default[2].get("style")
            if isinstance(name, str) and name in self.styles:
                fields, origin = self.flat_style(name)
                line.pop("style", None)
                for k, v in fields.items():
                    self._fill(line, k, v, f"{path}.{k}", origin[k])
                styled.add(path)
        for key, (src, dpath, body) in self.defaults.items():
            for path, item in _targets(data, key):
                for k, v in body.items():
                    if k == "style" and path in styled:
                        continue
                    self._fill(item, k, v, f"{path}.{k}", (src, f"{dpath}.{k}"))

    def _fill(self, item: dict, key: str, value: Any, path: str, origin: tuple[int, str]) -> None:
        """Set `key` unless the item already has it; dicts fill field by field."""
        if key not in item:
            if key == "emphasis" and _LINE_EMPHASIS.fullmatch(path) and _unmarked(item.get("text")):
                return  # #60: inherited emphasis is inert on a line with no `*marked*` words
            item[key] = copy.deepcopy(value)
            self.smap.entries[path] = origin
        elif isinstance(item[key], dict) and isinstance(value, dict):
            for k, v in value.items():
                self._fill(item[key], k, v, f"{path}.{k}", (origin[0], f"{origin[1]}.{k}"))


_LINE_EMPHASIS = re.compile(r"text\[\d+\]\.lines\[\d+\]\.emphasis")


def _unmarked(text: Any) -> bool:
    """True for a string line text with no `*spans*`. Non-strings and unclosed `*` stay as written."""
    if not isinstance(text, str):
        return False
    from montaj.render import glyphs

    try:
        return not any(glyphs.parse_emphasis(text)[1])
    except ValueError:
        return False


def _deep_override(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_override(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _targets(data: Any, key: str) -> list[tuple[str, dict]]:
    """Dicts a `defaults:` key names: `shots` = every shot, `text.lines` = every line of every block."""
    cur: list[tuple[str, Any]] = [("", data)]
    for part in key.split("."):
        nxt: list[tuple[str, Any]] = []
        for p, node in cur:
            if not isinstance(node, dict) or part not in node:
                continue
            child = node[part]
            cp = f"{p}.{part}" if p else part
            if isinstance(child, list):
                nxt.extend((f"{cp}[{i}]", it) for i, it in enumerate(child))
            else:
                nxt.append((cp, child))
        cur = nxt
    return [(p, n) for p, n in cur if isinstance(n, dict)]


# --- $param substitution -------------------------------------------------------------------


def _substitute(node: Any, vals: dict[str, Any], path: str) -> Any:
    if isinstance(node, dict):
        return {k: _substitute(v, vals, f"{path}.{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, vals, f"{path}[{i}]") for i, v in enumerate(node)]
    if not isinstance(node, str) or "$" not in node:
        return node
    m = _ARITH.match(node.strip())
    if m:
        name, sign, num, unit = m.groups()
        _known(name, vals, path)
        return _add_offset(vals[name], sign, float(num), unit, path, node)
    whole = _PARAM.fullmatch(node.strip())
    if whole:
        _known(whole.group(1), vals, path)
        return copy.deepcopy(vals[whole.group(1)])

    def sub(mm: re.Match) -> str:
        _known(mm.group(1), vals, path)
        return str(vals[mm.group(1)])

    return _PARAM.sub(sub, node)


def _known(name: str, vals: dict[str, Any], path: str) -> None:
    if name not in vals:
        raise _err(path, f"unknown param ${name}", f"params: {', '.join(vals) or 'none'}")


def _num(x: float) -> str:
    return format(x, ".10g")


def _add_offset(value: Any, sign: str, num: float, unit: str, path: str, raw: str) -> str:
    """`$at+1.2s` with at = `5s` -> `6.2s`; at = `ch1` -> `ch1+1.2s`; at = `ch1+1s` -> `ch1+2.2s`."""
    if not isinstance(value, str):
        raise _err(path, f"cannot do arithmetic on {value!r} in {raw!r}", "pass a duration or marker like 5s or ch1")
    value = value.strip()
    delta = num if sign == "+" else -num
    d = _DUR.match(value)
    if d:
        if d.group(2) != unit:
            raise _err(path, f"mixed units {value} and {unit} in {raw!r}", "use one unit")
        total = float(d.group(1)) + delta
        if total < 0:
            raise _err(path, f"{raw!r} with {value} is negative", "durations cannot be negative")
        return _num(total) + unit
    o = _OFFSET.match(value)
    if o:
        base, osign, onum, ounit = o.groups()
        if ounit != unit:
            raise _err(path, f"mixed units {value} and {unit} in {raw!r}", "use one unit")
        total = (float(onum) if osign == "+" else -float(onum)) + delta
        return base if total == 0 else f"{base}{'+' if total > 0 else '-'}{_num(abs(total))}{unit}"
    return f"{value}{sign}{_num(num)}{unit}"


# --- entry points --------------------------------------------------------------------------


def expand_spec(
    data: dict, text: str, spec_path: Path, builtin_styles: tuple[str, ...]
) -> tuple[dict, SourceMap | None]:
    """The plain spec for `data` loaded from `spec_path`, plus the map back to the author's keys.

    A spec with no reuse keys comes back unchanged with no map.
    """
    if not any(k in data for k in REUSE_KEYS):
        return data, None
    ex = _Expander(spec_path, builtin_styles)
    data = copy.deepcopy(data)
    ex.load(data, text)
    ex.check_style_names()
    ex.use(data)
    ex.apply(data)
    for k in REUSE_KEYS:
        data.pop(k, None)
    return data, ex.smap


def expand_file(path: Path | str, builtin_styles: tuple[str, ...]) -> dict:
    """Expanded spec of a file, for `montaj validate --expand`. Raises ExpandError."""
    path = Path(path)
    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise _err(".", "spec must be a mapping", "start with video:")
    out, _ = expand_spec(data, text, path, builtin_styles)
    return out
