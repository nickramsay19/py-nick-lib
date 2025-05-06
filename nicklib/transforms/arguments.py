"""Utility function transformers for function construction

Here, the term "transformer" denotes a operation on a function,
and in Python, is typically implemented as a decorator, or a
decorator factory. This module defines such function transformers 
for the  purposes of leveraging specific manipulations on function 
input or output whilst maintaining code brevity.

Typical usage example:

    @take_args_as_list()
    def f(nums: list[int]) -> int:
        return sum(nums)

    f(1, 2, 3) # 6

    @validate_args(lambda x: len(x) == 1)
    def g(a=0, b=0, c=0):
        return a + b - c

    g(a=1)           # AssertionError
    g(a=1, b=2)      # 3
    g(a=1, b=2, c=3) # AssertionError
    g(a=1, c=3)      # -2
    
"""
from __future__ import annotations
from typing import Callable, Any
import functools

type _Args = tuple[list[Any, ...], dict[str, Any]]
type _ArgMap = Callable[_Args, _Args]

# give a function transformerDecorator factory for functions to apply an arg transform on each call."""
def map_arguments(arg_transform: Callable[tuple[list, dict], tuple[list, dict]]) -> Callable:
    """Gives a transformer whose output functions' recieved input arguments are the result of the given argument mapping over the input arguments provided"""
    def _decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args, **kwargs) -> any:
            t_args, t_kwargs = arg_transform(*args, **kwargs)
            return func(*t_args, **t_kwargs)
        return _wrapper
    return _decorator

def take_args_as_list(pos: int = 0): # the factory
    """Gives a transformer whose output function's input arguments are all recieved in one list argument"""
    def take_args_as_list_decorator(func): # the actual decorator 
        @functools.wraps(func) # ensures decorator wrapping preserves original name
        def wrapper(*args, **kwargs):
            # extract only the desired list args
            normal_args = args[:pos]
            list_args = args[pos:]
            
            L: list[any] = list(list_args) # convert args tuple to list
            
            # check if we were passed a regular list
            if len(list_args) == 1 and type(list_args[0]) == list:
                # revert L back into the passed list
                L = list_args[0]
            
            return func(*normal_args, L, **kwargs)
        return wrapper
    return take_args_as_list_decorator

def validate_args(arg_validator: Callable) -> Callable:
    """Gives a transformer whose output function's assert conformance to the provided argument validator predicate before execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # an arg_validator can raise an error by itself, or return a BaseException for us to throw for it 
            validation = arg_validator(*args, **kwargs)
            if validation != None:
                raise validation from validation

            return func(*args, **kwargs)
        return wrapper
    return decorator
