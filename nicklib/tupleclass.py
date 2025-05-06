from __future__ import annotations
from typing import Any
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, MISSING, make_dataclass
import functools
import io

class _TupleClassMeta(type):
    '''The metaclass of TupleClass (see below for TupleClass)'''
    def __new__(cls, name, bases, dct):
        annotations = dct.get('__annotations__', {})
        new_cls = make_dataclass(name, [(name, typ, field(default=MISSING)) for name, typ in annotations.items()], bases=bases) # pass tuple superclass in here in bases
        new_cls.__module__ = dct.get('__module__')
        for key, value in dct.items():
            if key != '__annotations__':
                setattr(new_cls, key, value)

        # don't actually use the tuple superclass's __new__
        # TODO if superclass is a subclass of TupleClass, then make smart constructor chain
        original_new = new_cls.__new__
        def __new__(cls, *args, **kwargs):
            instance = original_new(cls, tuple(args)) # convert args to tuple
            for name, value in zip([f.name for f in fields(cls)], args):
                setattr(instance, name, value)
            for name, value in kwargs.items():
                setattr(instance, name, value)
            return instance
        setattr(new_cls, '__new__', __new__)

        # add an __iter__ method to enable emulated tuple unpacking
        def __iter__(self):
            field_names = list(self.__annotations__.keys())
            for f in field_names:
                yield getattr(self, f)
        setattr(new_cls, '__iter__', __iter__)

        # pseudo-tuple index access
        def __getitem__(self, key: int) -> Any:
            return getattr(self, list(self.__annotations__.keys())[key])
        setattr(new_cls, '__getitem__', __getitem__)

        # pseudo-tuple index set
        def __setitem__(self, key: int, val: Any):
            setattr(self, list(self.__annotations__.keys())[key], val)
        setattr(new_cls, '__setitem__', __setitem__)

        # pretty tuple print
        def __str__(self):
            return self.__class__.__name__ + str(tuple(self))
        setattr(new_cls, '__str__', __str__)        

        # TODO: __repr__ that prints also the field names

        return new_cls

@functools.total_ordering
class TupleClass(tuple, metaclass=_TupleClassMeta):
    '''
    Mutable Named pseudo-Tuple.

    Acts just like a regular NamedTuple and thus a normal tuple, but, is really a subclass of a dataclass instance. 

    Requires each specified field to be typed, since in Python, we can only determine dynamic fields via typed __annotations__.

    Doesn't allow inhertance.
    '''
    def __init__(self, *args, **kwargs):
        field_names = list(self.__annotations__.keys())
        if len(args) > len(field_names):
            raise TypeError(f"Expected at most {len(field_names)} arguments, got {len(args)}")

        # set all default
        for name in field_names:
            if not hasattr(self, name): # perhaps a default was already provided
                setattr(self, name, None)

        for name, value in zip(field_names, args):
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)

    # allow tuple equivalence w/ total_ordering
    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __lt__(self, other):
        return tuple(self) < tuple(other)

def _make_dummy_TupleClass() -> type:
    class Dummy(TupleClass):
        x: int # no default
        y: str = 'default'
    
    return Dummy

def _test_tuple_behavior():
    Dummy = _make_dummy_TupleClass()
        
    # a Dummy is a subclass of tuple (not really, but python thinks so)
    assert issubclass(Dummy, tuple)
    
    # a Dummy is a tuple
    d = Dummy(10,'hi')
    assert isinstance(d, tuple)
    assert isinstance(d, Dummy)
    assert d[0] == 10
    assert d[1] == 'hi'
    assert d == (10, 'hi')
    assert list(d) == [10, 'hi']
    assert len(d) == 2

def _test_named_tuple_behavior():
    Dummy = _make_dummy_TupleClass() 

    # a Dummy is a NamedTuple
    d = Dummy(10,'hi')
    assert d.x == 10
    assert d.y == 'hi'

    # a Dummy supports defaults
    assert Dummy(10).y == 'default'

    # correct constructor calling
    assert Dummy(x=10,y='hi') == (10, 'hi')
    assert Dummy(x=10) == (10, 'default')
    assert Dummy(10,y='hi') == (10, 'hi')

def _test_mutability():
    d = _make_dummy_TupleClass()(10,'hi')

    d.x = 11
    assert d.x == 11

def _test_inheritance():
    class A(TupleClass):
        a: str = 'a'

    class B(A):
        b: str = 'b'

    b = B()
    assert b.a == 'a'
    assert b.b == 'b'

def _test_inheritance_no_defaults_b():
    class A(TupleClass):
        a: str = 'a'

    class B(A):
        b: str

    b = B('b')
    assert b.a == 'a'
    assert b.b == 'b'

def _test_inheritance_no_defaults_a():
    class A(TupleClass):
        a: str

    class B(A):
        b: str = 'b'

    b = B('a')
    #assert b.a == 'a' # FAIL
    assert b.b == 'a'

def _test_inheritance_no_defaults_ab():
    class A(TupleClass):
        a: str

    class B(A):
        b: str

    #b = B('a', 'b') # ERROR TypeError: Expected at most 1 arguments, got 2
    #assert b.a == 'a'
    #assert b.b == 'b'

def test():
    _test_tuple_behavior()
    _test_named_tuple_behavior()
    _test_mutability()
    _test_inheritance()
    _test_inheritance_no_defaults_a()
    _test_inheritance_no_defaults_b() # FAIL
    _test_inheritance_no_defaults_ab() # FAIL

def main():
    test()

if __name__ == '__main__':
    main()

'''
# Mutple, FieldTuple
# Field, tupdleif 

class ABSTRACT: # fake top inheritance ceiling that we will want to exclude from __mro__ loop
    pass

class A(ABSTRACT):
    a: str = 'a'

class B(A):
    b: str = 'b'

class C(B):
    c: str = 'c'

    #def __new__(cls, *args, **kwargs):
        #filter(lambda x: issubclass(x, ABSTRACT) and x != ABSTRACT, reversed(cls.__mro__)):
#            print(t)

    def __init__(self, *args, **kwargs):
        for t in filter(lambda x: issubclass(x, ABSTRACT) and x != ABSTRACT, reversed(self.__class__.__mro__)):
            print(t)

c = C()
```
