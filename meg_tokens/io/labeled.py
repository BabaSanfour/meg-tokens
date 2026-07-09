"""Named-dimension adapters for the array-plus-sidecar contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .contract import load_array, save_array


_NAME_KEY = "dataarray_name"


def save_dataarray(
    path: str | Path,
    data: xr.DataArray,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save an xarray object without changing the `.npy`/JSON contract."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}")
    if len(set(data.dims)) != len(data.dims):
        raise ValueError(f"DataArray dimensions must be unique: {data.dims}")
    unsupported_coords = sorted(set(data.coords) - set(data.dims))
    if unsupported_coords:
        raise ValueError(
            "Only one-dimensional coordinates named after their dimensions can "
            f"be stored in the npy+JSON contract; unsupported: {unsupported_coords}"
        )

    coords: dict[str, Any] = {}
    for dim, size in data.sizes.items():
        if dim not in data.coords:
            continue
        coordinate = data.coords[dim]
        if coordinate.dims != (dim,) or coordinate.size != size:
            raise ValueError(f"Coordinate '{dim}' must be one-dimensional over its matching dimension")
        coords[dim] = coordinate.values

    attrs = dict(data.attrs)
    attrs.update(metadata or {})
    if data.name is not None:
        attrs[_NAME_KEY] = str(data.name)
    return save_array(
        path,
        data.values,
        dims=data.dims,
        coords=coords,
        metadata=attrs,
    )


def load_dataarray(path: str | Path) -> xr.DataArray:
    """Load a sidecar-backed array as an xarray object with validated labels."""
    loaded = load_array(path, require_sidecar=True)
    sidecar = loaded.metadata
    dims = tuple(sidecar.get("dims", ()))
    if len(dims) != loaded.data.ndim or len(set(dims)) != len(dims):
        raise ValueError(f"Invalid dimensions in array sidecar for {path}: {dims}")

    shape = tuple(sidecar.get("shape", ()))
    if shape and shape != loaded.data.shape:
        raise ValueError(
            f"Sidecar shape {shape} does not match array shape {loaded.data.shape} for {path}"
        )

    coords = {}
    raw_coords = sidecar.get("coords", {})
    if not isinstance(raw_coords, Mapping):
        raise ValueError(f"Sidecar coords must be a mapping for {path}")
    for axis, dim in enumerate(dims):
        if dim not in raw_coords:
            continue
        values = np.asarray(raw_coords[dim])
        if values.ndim != 1 or values.size != loaded.data.shape[axis]:
            raise ValueError(
                f"Coordinate '{dim}' has length {values.size}; "
                f"expected {loaded.data.shape[axis]} for {path}"
            )
        coords[dim] = values

    raw_attrs = sidecar.get("metadata", {})
    if not isinstance(raw_attrs, Mapping):
        raise ValueError(f"Sidecar metadata must be a mapping for {path}")
    attrs = dict(raw_attrs)
    name = attrs.pop(_NAME_KEY, None)
    return xr.DataArray(
        loaded.data,
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=name,
    )
