import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any


def _encode_nonstandard_scalar(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def encode_tuple(array: Iterable[int | float | str]) -> str:
    return json.dumps(tuple(array), default=_encode_nonstandard_scalar)


def decode_json_array(json_array: str) -> tuple[int | float | str, ...]:
    return tuple(json.loads(json_array))

def encode_datetime(dt: datetime) -> str:
    return dt.isoformat()

def decode_datetime_string(json_string: str) -> datetime:
    return datetime.fromisoformat(json_string)
