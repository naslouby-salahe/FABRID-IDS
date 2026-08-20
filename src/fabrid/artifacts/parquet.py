from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, ClassVar, get_args, get_origin, get_type_hints

import polars as pl
from pydantic import BaseModel

_PYTHON_TO_POLARS: dict[type[object], pl.DataType] = {
    str: pl.String(),
    int: pl.Int64(),
    float: pl.Float64(),
    bool: pl.Boolean(),
}


def _field_dtype(annotation: object) -> pl.DataType:
    resolved = annotation
    while True:
        origin = get_origin(resolved)
        if origin is Annotated:
            resolved = get_args(resolved)[0]
            continue
        if origin is None:
            break
        args = get_args(resolved)
        if len(args) == 2 and type(None) in args:
            resolved = args[0] if args[1] is type(None) else args[1]
            continue
        raise TypeError(f"unsupported annotated column {resolved!r}")
    if not isinstance(resolved, type):
        raise TypeError(f"no polars dtype mapping for annotation {resolved!r}")
    for python_type, dtype in _PYTHON_TO_POLARS.items():
        if issubclass(resolved, python_type):
            return dtype
    raise TypeError(f"no polars dtype mapping for annotation {resolved!r}")


def _polars_schema(model_type: type) -> pl.Schema:
    if is_dataclass(model_type):
        annotations = {
            name: annotation
            for name, annotation in get_type_hints(model_type).items()
            if get_origin(annotation) is not ClassVar
        }
    elif issubclass(model_type, BaseModel):
        annotations = {name: field.annotation for name, field in model_type.model_fields.items()}
    else:
        raise TypeError(f"unsupported parquet model type {model_type!r}")
    return pl.Schema((name, _field_dtype(annotation)) for name, annotation in annotations.items())


def write_parquet_models(path: Path, models: tuple[object, ...]) -> None:
    if not models:
        raise ValueError("cannot write an empty parquet artifact without records")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _polars_schema(type(models[0]))
    rows: list[dict[str, object]] = []
    for model in models:
        if is_dataclass(model):
            row: dict[str, object] = {}
            for field in fields(model):
                value = getattr(model, field.name)
                row[field.name] = value.value if isinstance(value, Enum) else value
            rows.append(row)
        elif isinstance(model, BaseModel):
            rows.append(model.model_dump(mode="json"))
        else:
            raise TypeError(f"unsupported parquet record type {type(model)!r}")
    frame = pl.DataFrame(rows, schema=schema)
    frame.write_parquet(path)


def read_parquet_models[ModelT](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    frame = pl.read_parquet(path)
    rows = frame.to_dicts()
    if is_dataclass(model):
        return tuple(model(**row) for row in rows)
    if issubclass(model, BaseModel):
        return tuple(model.model_validate(row) for row in rows)
    raise TypeError(f"unsupported parquet model type {model!r}")


def write_parquet_frame(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
