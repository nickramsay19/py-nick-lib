from __future__ import annotations
from typing import Union, Callable, Literal
from collections.abc import Sequence, Iterator
import itertools
import functools
import operator
import unittest

class Vec[T]:
    _components: list[T]

    def __init__(self, *args):
        self._components = args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self._components = [*args[0]]
        else:
            self._components = [*args]

    def __getitem__(self, idx: int) -> T:
        return self._components[idx]

    def __setitem__(self, idx: int, val: T):
        self._components[idx] = val

    def __len__(self):
        return len(self._components)

    def __repr__(self):
        return f"Vec({', '.join(map(str, self._components))})"

    def __iter__(self) -> Iterator[T]:
        return iter(self._components)

    def _expand_other(self, other: Union[Vec[T],T]) -> Vec[T]:
        """Utility method to return the given value, or, if it is a scalar, return a Vec with all n components the value of that scalar"""
        if isinstance(other, Vec):
            return other
        elif isinstance(other, Sequence):
            return Vec(other)
        else:
            return Vec([other]*len(self._components))

    @staticmethod
    def _accept_expanded_other(f: Callable) -> Callable:
        """Utility decorator that makes any method taking an 'other' object accept either a Vec of the same degree or a scalar"""
        @functools.wraps(f)
        def _f(self, *args, **kwargs) -> Any:
            return f(self, self._expand_other(args[0]), *args[1:], **kwargs)
        return _f

    @staticmethod
    def _fmap(f: Callable[T,T]) -> Callable[T,Vec[T]]:
        def _map(vec: Vec[T]) -> Vec[T]: 
            return Vec([*map(f, vec._components)])
        return _map

    @staticmethod
    def _fmap2(f: Callable[T,T]) -> Callable[Vec[T],Callable[Vec[T],Vec[T]]]:
        def _map2_1(vec1: Vec[T]) -> Vec[T]: 
            def _map2_2(vec2: Vec[T]) -> Vec[T]: 
                return Vec([*itertools.starmap(f, zip(vec1._components, vec2._components))])
            return _map2_2
        return _map2_1

    def magnitude(self) -> T:
        return sum(self**2)

    def direction(self) -> Vec[T]:
        return self / self.magnitude()

    def __neg__(self) -> Vec[T]:
        return self._fmap(operator.neg)(self)

    @_accept_expanded_other
    def __add__(self, other: Vec) -> Vec:
        return self._fmap2(operator.add)(self)(other)

    @_accept_expanded_other
    def __sub__(self, other: Vec) -> Vec:
        return self._fmap2(operator.add)(self)(-other)

    @_accept_expanded_other
    def __mul__(self, other: Vec) -> float:
        return sum(self._fmap2(operator.mul)(self)(other)._components)

    def __pow__(self, p: int) -> float:
        return sum(self._fmap(lambda x: x**p)(self)._components)

    @_accept_expanded_other
    def __iadd__(self, other: Vec):
        self._components = self.__add__(other)._components
        return self

    def __eq__(self, other: Vec):
        return self._fmap2(operator.eq)(self)(other)

    def __ne__(self, other: Vec):
        return not self.__eq__(other)

class VecXZ[T](Vec[T]):
    '''for 2d position, force, velocity, acceleration vectors'''
    @property
    def x(self) -> T:
        return self._components[0]

    @property
    def z(self) -> T:
        return self._components[1]

class VecX[T](Vec[T]):
    '''for 2d line length vectors'''
    @property
    def x(self) -> T:
        return self._components[0]

class VecY[T](Vec[T]): 
    '''for 2d angle vectors'''
    @property
    def y(self) -> T:
        return self._components[0]

class VecXYZ[T](Vec[T]):
    '''for 3d position, force, velocity, acceleration vectors'''
    @property
    def x(self) -> T:
        return self._components[0]

    @property
    def y(self) -> T:
        return self._components[1]

    @property
    def z(self) -> T:
        return self._components[3]

class TestVec(unittest.TestCase):
    def test_constructor(self):
        v1 = Vec([1,2,3])
        v2 = Vec((1,2,3))
        v3 = Vec(1,2,3)
        
        self.assertEqual(v1._components, v2._components)
        self.assertEqual(v1._components, v3._components)

    def test_add(self):
        v = Vec(1,2,3)
        u = Vec(1,2,3)
        self.assertEqual((v + u)._components, [2,4,6])

if __name__ == '__main__':
    unittest.main()
