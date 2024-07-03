from functools import wraps


class Library(object):
    constexpr = int
    autotune = lambda *args, **kwargs: wraps  # noqa: E731
    Config = lambda *args, **kwargs: None  # noqa: E731
    jit = wraps
    heuristics = lambda *args, **kwargs: wraps  # noqa: E731
