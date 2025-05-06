from __future__ import annotations
from typing import Any, Callable
import functools

_INCREMENT_ARG_TRANSFORM = lambda x: x + 1

def _make_reference(val):
    ref = [val]
    def getter():
        return ref[0]
    def setter(new_val):
        ref[0] = new_val
    return getter, setter

class _WrappedObject:
    """Wrap atomics and sentinals into a pseudo pass by reference object."""
    def __init__(self, val: any):
        self._val: List[any] = [val]

    def _get(self) -> any:
        return self._val[0]
    
    def _set(self, val: any) -> None:
        self._val = [val]

    def GetGetter(self) -> Callable[[], any]:
        return lambda: self._get()
        
    def GetSetter(self) -> Callable[[any], None]:
        return lambda val: self._set(val)
    
    def GetModifier(self, transform: Callable[[any], any]) -> Callable[[any], None]:
        return lambda: self._set(transform(self._val[0]))
    
    def GetValue(self) -> any:
        v = self._get()
        return v

def limit_calls(max_calls: int = 1, then = lambda *args, **kwargs: None) -> Callable:
    """A function decorator that limits the total number of calls executed of a function. Useful for ensuring that test cases are ran only once."""
    def decorator(func):
    
        # count function calls, value must be wrapped so it can be modified
        calls = _WrappedObject(0) 
        get_calls = calls.GetGetter()
        inc_calls = calls.GetModifier(_INCREMENT_ARG_TRANSFORM)
        
        @functools.wraps(func)
        def wrapper(get_calls_get, inc_calls, *args, **kwargs):
            if get_calls() < max_calls:
                inc_calls()
                return func(*args, **kwargs)
            else:
                return then(*args, **kwargs)
        return lambda *args, **kwargs: wrapper(get_calls, inc_calls, *args, **kwargs)
    return decorator
