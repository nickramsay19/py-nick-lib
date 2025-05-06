from __future__ import annotations
from typing import Any, Iterable

def attr_unpack(obj: object, *args: list[str]) -> list[object]:
    return [getattr(obj, arg) for arg in args]

def iter_attr_unpack(iterable: iter, *args: list[str]) -> list[object]:
    return [attr_unpack(obj, *args) for obj in iterable]
