# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Class-map discovery for YOLO datasets (dataset.yaml / data.yaml / classes.txt)
and generic classes-list loading.

The fallback YAML parser is ported from remove_class.py so PyYAML stays optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

YAML_PRIORITY = ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")
TXT_NAMES = ("classes.txt", "obj.names", "classes.names")


@dataclass
class ClassMap:
    names: list[str] = field(default_factory=list)
    source: Path | None = None
    kpt_shape: list[int] | None = None
    problems: list[str] = field(default_factory=list)
    kind: str | None = None        # "yaml" | "txt" (found in dataset) | "user" (loaded via Load Classes) | None


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
        return v[1:-1]
    return v


def _parse_yaml_fallback(text: str) -> tuple[list[str] | None, int | None]:
    lines = text.splitlines()
    nc = None
    for line in lines:
        s = line.strip()
        if s.startswith("nc:") and not line.startswith((" ", "\t")):
            try:
                nc = int(s.split(":", 1)[1].split("#")[0].strip())
            except ValueError:
                nc = None
    idx = next((i for i, l in enumerate(lines) if l.strip().startswith("names:") and not l.startswith((" ", "\t"))), None)
    if idx is None:
        return None, nc
    value = lines[idx].split(":", 1)[1].split("#")[0].strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return ([_unquote(p) for p in inner.split(",")] if inner else []), nc
    mapping: dict[int, str] = {}
    listed: list[str] = []
    for child in lines[idx + 1:]:
        if not child.strip():
            continue
        if not child.startswith((" ", "\t")):
            break
        s = child.strip()
        if s.startswith("- "):
            listed.append(_unquote(s[2:]))
        elif ":" in s:
            k, v = s.split(":", 1)
            if k.strip().isdigit():
                mapping[int(k.strip())] = _unquote(v.split("#")[0])
    if mapping:
        if set(mapping) != set(range(max(mapping) + 1)):
            return None, nc
        return [mapping[i] for i in range(len(mapping))], nc
    return (listed or None), nc


def parse_yaml_class_map(path: Path) -> ClassMap:
    cm = ClassMap(source=path, kind="yaml")
    text = path.read_text(encoding="utf-8", errors="replace")
    names, nc = None, None
    if yaml is not None:
        try:
            doc = yaml.safe_load(text) or {}
            raw = doc.get("names")
            if isinstance(raw, dict):
                keys = sorted(int(k) for k in raw)
                if keys == list(range(len(keys))):
                    names = [str(raw[k] if k in raw else raw[str(k)]) for k in keys]
            elif isinstance(raw, list):
                names = [str(n) for n in raw]
            nc = doc.get("nc")
            ks = doc.get("kpt_shape")
            if isinstance(ks, list):
                cm.kpt_shape = ks
        except Exception as exc:
            cm.problems.append(f"YAML parse error: {exc}")
    if names is None:
        names, nc2 = _parse_yaml_fallback(text)
        nc = nc if nc is not None else nc2
        if "kpt_shape" in text:
            cm.kpt_shape = [0, 0]  # only presence matters for rendering
    if names is None:
        cm.problems.append(f"Could not parse 'names' from {path.name}")
        return cm
    cm.names = [n.strip() for n in names]
    if any(not n for n in cm.names):
        cm.problems.append("Class map contains an empty class name")
    if len(set(cm.names)) != len(cm.names):
        cm.problems.append("Class map contains duplicate class names")
    if nc is not None and int(nc) != len(cm.names):
        cm.problems.append(f"nc={nc} but names has {len(cm.names)} entries")
    return cm


def load_classes_file(path: str | Path) -> list[str]:
    """User-provided truth list: .txt (one per line) or .yaml (names:)."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        return parse_yaml_class_map(path).names
    return [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]


def txt_class_map(path: Path, kind: str = "txt") -> ClassMap:
    names = load_classes_file(path)
    cm = ClassMap(names=names, source=Path(path), kind=kind)
    if not names:
        cm.problems.append(f"{Path(path).name} is empty")
    if len(set(names)) != len(names):
        cm.problems.append(f"{Path(path).name} contains duplicate class names")
    return cm


def _yaml_in(base: Path) -> ClassMap | None:
    for name in YAML_PRIORITY:
        p = base / name
        if p.is_file():
            return parse_yaml_class_map(p)
    yamls = sorted(p for p in base.glob("*.y*ml") if p.is_file())
    for p in yamls:
        try:
            if "names" in p.read_text(encoding="utf-8", errors="replace"):
                cm = parse_yaml_class_map(p)
                if len(yamls) > 1:
                    cm.problems.append(f"Multiple YAML files found; using {p.name}")
                return cm
        except OSError:
            continue
    return None


def find_class_map(root: Path, label_dirs: tuple[Path, ...] = ()) -> ClassMap:
    """
    YOLO class-map resolution order:
      1. dataset.yaml / data.yaml (any *.yaml with `names`) in the dataset root
      2. classes.txt / obj.names in root, root/labels, or any <split>/labels
      3. a YAML in the parent folder (user opened e.g. root/train directly)
    Returns an empty ClassMap (kind=None) when nothing is found -> the GUI asks for classes.txt.
    """
    root = Path(root)
    cm = _yaml_in(root)
    if cm is not None:
        return cm
    for d in (root, root / "labels", *label_dirs):
        for name in TXT_NAMES:
            p = d / name
            if p.is_file():
                return txt_class_map(p)
    cm = _yaml_in(root.parent)
    if cm is not None:
        cm.problems.append(f"Using {cm.source.name} from the parent folder")
        return cm
    return ClassMap(problems=["No dataset.yaml / data.yaml / classes.txt found"])
