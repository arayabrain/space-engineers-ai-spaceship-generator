import json
import pickle
from typing import IO, Any


# Adapted from https://stackoverflow.com/questions/18478287/making-object-json-serializable-with-regular-encoder/18561055#18561055
class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        return {'_python_object': pickle.dumps(obj).decode('latin1')}


def _as_python_object(dct):
    try:
        return pickle.loads(dct['_python_object'].encode('latin1'))
    except KeyError:
        return dct


def json_dump(obj: Any,
              fp: IO[str]) -> None:
    """Dump a Python object to file as JSON.
    Non-serializable Python objects are first converted to `str` of `pickle`.

    Args:
        obj (Any): The Python object.
        fp (IO[str]): The file pointer.
    """
    json.dump(obj=obj,
              fp=fp,
              cls=PythonObjectEncoder)


def json_load(fp: IO[str]) -> Any:
    """Load a Python object from JSON file.
    Non-serializable Python objects are loaded from their `pickle`d `str` representation.

    Args:
        fp (IO[str]): The file pointer.

    Returns:
        Any: The Python object.
    """
    return json.load(fp=fp,
                     object_hook=_as_python_object)


def json_dumps(obj: Any) -> str:
    """Dump a Python object to JSON `str`.
    Non-serializable Python objects are first converted to `str` of `pickle`.

    Args:
        obj (Any): The Python object.

    Returns:
        str: The JSON `str`.
    """
    return json.dumps(obj=obj,
                      cls=PythonObjectEncoder)


def json_loads(s: str) -> Any:
    """Load a Python object from JSON `str`.
    Non-serializable Python objects are loaded from their `pickle`d `str` representation.

    Args:
        s (str): The JSON `str`.

    Returns:
        Any: The Python object.
    """
    return json.loads(s=s,
                      object_hook=_as_python_object)
