from __future__ import annotations
from typing import Callable, Any
from functools import reduce

'''def compose(*funcs: Callable) -> Callable:
    g = lambda x: x # identity
    for f in funcs:
        g = lambda x: f(g(x))

    return g'''

type F[A,B] = Callable[A,B]

_compose_2[A,B]: Callable[(F[A,B], F[A,B]), F[A,B]] = lambda f, g: lambda x: f(g(x))
compose[T]: Callable[(F[T,T], ...), F[T,T]] = lambda *fs: reduce(_compose_2, reversed(fs))

'''def extract_map(f: Callable[[int,int] -> Any]) -> Callable[int,np.ndarry]:
    """generates a function that extracts a bounded grid from a coordinate to value function"""
    def inner(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        extract = np.zeros((y1-y0, x1-x0), dtype=np.double)

        for y in range(y0, y1):
            for x in range(x0, x1):
                extract[y-y0][x-x0] = f(x, y)

        return extract

    return inner

def is_non_negative_int(v: str) -> bool:
    if v.isnumeric():
        x = int(v)
        return x >= 0
    return False'''

#checker: Callable = compose(lambda x: x.isnumeric()
'''checker: Callable = lambda x: reduce(lambda y, f: f(y), [
    lambda: x, 
    str.isnumeric,'''

