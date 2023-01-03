FILE_SCHEME = "file://"

def parse_uri(path):
    """Parse a path into a schema"""
    path = path.strip()
    scheme = None
    old_scheme = None
    if "://" not in path:
        return FILE_SCHEME, path
    elif path.startswith(FILE_SCHEME):
        scheme = FILE_SCHEME
    else:
        raise ValueError("Wrong path provided: %s" % path)
    scheme_prefix = scheme
    return scheme, path[len(scheme_prefix) :]