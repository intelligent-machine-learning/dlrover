def raise_not_impl(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__
        method_name = func.__name__
        raise NotImplementedError(f"{class_name} does not implement function {method_name}")

    return wrapper
