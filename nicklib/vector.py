from __future__ import annotations
from typing import Union, Callable, Literal
from collections.abc import Sequence, Iterator
import itertools
import functools
import operator
import unittest
from math import sqrt, atan2
import numpy as np
from tupleclass import TupleClass

def _make_property(index):
    def getter(_self):
        return _self._items[index]
    def setter(_self, value):
        _self._items[index] = value

    return property(getter, setter)

class _ArrayClass(TupleClass):
    """Implementation of TupleClass that uses a numpy array for storage.

    This is not a particularly useful class globally so it remains hidden. 
    However, it provides useful basic functionality for the Vec class.
    """

    # don't assign type to avoid it from being used by TupleClass
    _items = None

    def __new__(cls, *args, **kwargs):
        new_cls = super().__new__(cls)

        for i, key in enumerate(cls.__annotations__.keys()):
            default = getattr(cls, key, None)
            if not isinstance(default, property):
                if default != None:
                    setattr(cls, '_' + key + '_default', default)

                setattr(cls, key, _make_property(i))

        return new_cls

    def __init__(self, *args, **kwargs):
        # determine the type of the values
        annotation_vals = list(self.__annotations__.values())
        if len(annotation_vals) > 0:
            dtype = annotation_vals[0]
            self._items = np.zeros(len(self.__annotations__), dtype=dtype)
            super().__init__(*args, **kwargs)

            for name in self.__annotations__:
                if (default := getattr(self.__class__, '_' + name + '_default', None)) != None:
                    setattr(self, name, default)
        else:
            self._items = np.array(args, **kwargs)

    def __getitem__(self, idx: int) -> T:
        return self._items[idx]

    def __setitem__(self, idx: int, val: T):
        self._items[idx] = val

    def __len__(self):
        return len(self._items)

    def __tuple__(self) -> tuple[T]: # necessery for unpacking
        return tuple(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.__tuple__())

class Vec(_ArrayClass):
    @staticmethod
    def _accept_expanded_other(f: Callable) -> Callable:
        """Utility decorator that makes any method taking an 'other' object accept either a Vec of the same degree or a scalar"""
        @functools.wraps(f)
        def _f(_self, *args, **kwargs) -> Any:
            other = args[0]

            if isinstance(other, Vec):
                expanded_other = other
            elif isinstance(other, Sequence):
                expanded_other = _self.__class__(other)
            else:
                expanded_other = _self.__class__(*[other]*len(_self._items))

            return f(_self, expanded_other, *args[1:], **kwargs)
        return _f

    @staticmethod
    def _fmap(f: Callable[T,T]) -> Callable[T,Vec[T]]:
        def _map(vec: Vec[T]) -> Vec[T]: 
            return vec.__class__(*map(f, vec._items))
        return _map

    @staticmethod
    def _fmap2(f: Callable[T,T]) -> Callable[Vec[T],Callable[Vec[T],Vec[T]]]:
        def _map2_1(vec1: Vec[T]) -> Vec[T]: 
            def _map2_2(vec2: Vec[T]) -> Vec[T]: 
                #return Vec([*itertools.starmap(f, zip(vec1._items, vec2._items))])
                return vec1.__class__(*itertools.starmap(f, zip(vec1._items, vec2._items)))
            return _map2_2
        return _map2_1

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self._items))})"

    def magnitude(self) -> T:
        return sqrt(sum(self**2))

    def direction(self) -> Vec[T]:
        return self / self.magnitude()

    def argument(self) -> T:
        # only works for vectors of length 2
        return atan2(*self._items[:2])

    def __neg__(self) -> Vec[T]:
        return self._fmap(operator.neg)(self)

    @_accept_expanded_other
    def __add__(self, other: Vec) -> Vec[T]:
        return self._fmap2(operator.add)(self)(other)

    @_accept_expanded_other
    def __sub__(self, other: Vec) -> Vec[T]:
        return self._fmap2(operator.add)(self)(-other)

    @_accept_expanded_other
    def __mul__(self, other: Vec) -> Vec[T]:
        return self._fmap2(operator.mul)(self)(other)

    def __rmul__(self, lhs: T) -> T:
        return self.__mul__(lhs)

    def __imul__(self, other: Vec) -> Vec[T]:
        self._items = self.__mul__(other)._items
        return self

    def __truediv__(self, rhs: T) -> T:
        return self._fmap(lambda lhs: lhs / rhs)(self)

    def __pow__(self, p: int) -> float:
        return self._fmap(lambda x: x**p)(self)

    @_accept_expanded_other
    def __iadd__(self, other: Vec):
        self._items = self.__add__(other)._items
        return self

    def __eq__(self, other: Vec):
        return self._fmap2(operator.eq)(self)(other)

    def __ne__(self, other: Vec):
        return not self.__eq__(other)

class TestVec(unittest.TestCase):
    def test_basic_array_class(self):
        class VecXY(_ArrayClass):
            x: float 
            y: float

        # test basic methods when all args provided
        v1 = VecXY(5.5, 1.2)
        assert v1.x == 5.5
        assert v1.y == 1.2
        assert v1[0] == 5.5
        assert v1[1] == 1.2
        assert len(v1) == 2

        # test basic methods when not all args provided
        # values should be initialised to 0.0
        v1 = VecXY(5.5)
        assert v1.x == 5.5
        print(v1.y)
        assert v1.y == 0.0
        assert v1[0] == 5.5
        assert v1[1] == 0.0
        assert len(v1) == 2

        # test that casting occurs
        v2 = VecXY(1) 
        assert v2.x == 1.0

    def test_default_value_array_class(self):
        class VecXY(_ArrayClass):
            x: float = 1.0
            y: float

        v1 = VecXY()
        assert v1.x == 1.0
        assert v1.y == 0.0
        assert v1[0] == 1.0
        assert v1[1] == 0.0
        assert len(v1) == 2

    def test_default_value_array_class_2(self):
        class VecXY(_ArrayClass):
            x: float
            y: float = 1.0

        v1 = VecXY()
        assert v1.x == 0.0
        assert v1.y == 1.0
        assert v1[0] == 0.0
        assert v1[1] == 1.0
        assert len(v1) == 2

    def test_basic_vec(self):
        v = Vec(1,2)
        assert v[0] == 1
        assert v[1] == 2
        assert len(v) == 2

        u = v + 1
        assert u[0] == 2

        assert (v * 2)[1] == 4

        # the original object is unchanged
        assert v[0] == 1
        assert v[1] == 2
        assert len(v) == 2

        v += 1
        assert v[0] == 2
        assert v[1] == 3
        assert len(v) == 2

    def test_typed_vec(self):
        class VecXY(Vec):
            x: float
            y: float

        v = VecXY(1,2)
        assert v.x == 1
        assert v[0] == 1
        assert v.y == 2
        assert v[1] == 2
        assert len(v) == 2

        u = v + 1
        assert u[0] == 2
        assert u.x == 2

        assert (v * 2)[1] == 4
        assert (v * 2).y == 4

        # the original object is unchanged
        assert v[0] == 1
        assert v.x == 1
        assert v[1] == 2
        assert v.y == 2
        assert len(v) == 2

        v += 1
        assert v[0] == 2
        assert v.x == 2
        assert v[1] == 3
        assert v.y == 3
        assert len(v) == 2

if __name__ == '__main__':
    unittest.main()
