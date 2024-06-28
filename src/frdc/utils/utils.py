from collections import namedtuple

Rect = namedtuple("Rect", ["x0", "y0", "x1", "y1"])


def map_nested(x, fn, type_atom, type_list):
    """Recursively applies a function to the data preserving the structure

    Args:
        x: The data to apply the function to.
        fn: The function to apply.
        type_atom: The type of the atom.
        type_list: The type of the list.
    """
    if isinstance(x, type_atom):
        return fn(x)
    elif isinstance(x, type_list):
        return [map_nested(item, fn, type_atom, type_list) for item in x]
    else:
        return x


def flatten_nested(x, type_list):
    """Flattens a nested list"""
    if not isinstance(x, type_list):
        return [x]
    return [
        item for sub in x for item in flatten_nested(sub, type_list=type_list)
    ]
