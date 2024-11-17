from __future__ import annotations
from typing import Any, Callable

def assert_error(exp_error: BaseException|type, func: Callable, *args, **kwargs) -> Any:
    """Debug assert that a call with provided args will result in the specified exception raised. Will raise an AssertionError. If exp_error is None, then assert that no error will be raised. For convinience, a successful call's value will be retured."""
    try: # call the given function, 
        return func(*args, **kwargs) # return its value in case the caller wants to do further checks
    except BaseException as e:
        if type(e) == exp_error or type(e) == type(exp_error): # caller expected this error, all is good
            return
        elif exp_error == None: # caller didn't except any error, hence None
            raise AssertionError(f"Function {func.__name__} raised {repr(e)}, but no error was expected.") from e
        else: # the caller didn't expect this error to be raised
            raise AssertionError(f"Function {func.__name__} raised {repr(e)}, but a {exp_error.__name__} was expected.") from e
