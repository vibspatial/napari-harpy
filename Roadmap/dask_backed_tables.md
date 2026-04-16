# Dask-Backed AnnData Tables in SpatialData Zarr

## Scope

This note summarizes local observations about:

- lazy `AnnData` tables with Dask-backed `.X`
- lazy Dask-backed feature matrices in `adata.obsm[...]`
- what happens when writing these tables to a SpatialData Zarr store
- what happens when reading them back with `anndata` and `spatialdata`

The focus is on practical behavior in the current project environment, not on API guarantees across all versions.

## Environment

Observed locally in the project `.venv`:

- `anndata 0.12.10`
- `spatialdata 0.7.2`

## Short Answer

- Yes: `AnnData` can hold Dask-backed `.X` and Dask-backed `obsm[...]` in memory.
- Yes: writing those tables to Zarr stores the underlying array data as Zarr-backed arrays on disk.
- No: the default `anndata.read_zarr(...)` and `spatialdata.read_zarr(...)` paths do not preserve lazy table matrices on read.
- Yes: `anndata.experimental.read_lazy(...)` can reopen a table Zarr group lazily, including `.X` and `obsm[...]`.
- No: the on-disk representation is not a persisted Dask graph. It is Zarr array storage that can later be reopened lazily.

## AnnData Observations

### In-memory support

Local experiments showed that `AnnData` accepts:

- dense NumPy arrays in `.X` and `.obsm`
- SciPy sparse matrices in `.obsm`
- Dask arrays in `.X` and `.obsm`

For `obsm`, the main built-in constraint is alignment with `n_obs` on axis 0.

Important nuance:

- `obsm` is not restricted to 2D by `AnnData`
- local tests accepted Dask arrays with shapes `(n_obs,)`, `(n_obs, n_features)`, and `(n_obs, 2, 2)`

Implication:

- if we want `obsm[feature_key]` to mean a strict feature matrix, we need to validate that contract ourselves
- we should not rely on `AnnData` to enforce `n_obs x n_features`

Relevant local source:

- [aligned_mapping.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/anndata/_core/aligned_mapping.py:70)
- [anndata.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/anndata/_core/anndata.py:1831)

### Writing with `adata.write_zarr(...)`

Observed locally:

- Dask-backed `.X` and Dask-backed `obsm[...]` can be written with `adata.write_zarr(...)`
- after writing, the in-memory `AnnData` object still held Dask arrays
- the write operation did not mutate the in-memory object into NumPy arrays

This means:

- writing evaluates data for persistence
- but it does not replace the in-memory lazy arrays with eager ones

### Reading with `anndata.read_zarr(...)`

Observed locally:

- Dask-backed dense arrays written to `.obsm` came back as eager `ndarray`
- Dask-backed `.X` also came back eager through the normal read path

So the default `read_zarr(...)` path materializes these dense arrays.

### Reading with `anndata.experimental.read_lazy(...)`

Observed locally:

- `read_lazy(...)` could reopen `.X` lazily
- `read_lazy(...)` could reopen `obsm[...]` lazily

This worked both for:

- a plain AnnData Zarr store
- a table group inside a SpatialData Zarr store

Practical consequence:

- the underlying storage is compatible with lazy reopening
- but we only get that laziness back if we use the experimental lazy reader

## SpatialData Observations

### In-memory table acceptance

`SpatialData` tables are plain `AnnData` objects validated through `TableModel`.

Observed locally:

- `TableModel.parse(...)` accepted Dask-backed `.X`
- `TableModel.parse(...)` accepted Dask-backed `obsm[...]`
- `SpatialData(tables={...})` preserved those Dask-backed arrays in memory

Relevant local source:

- [models.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/models/models.py:1073)
- [_elements.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_core/_elements.py:121)

### Writing with `SpatialData.write(...)` and `write_element(...)`

`SpatialData.write(...)` and `SpatialData.write_element(...)` both delegate table writing to the same table writer:

- `_write_element(...)` in SpatialData
- `write_table(...)` in `spatialdata._io.io_table`
- which then calls AnnData's `write_elem(...)`

Relevant local source:

- [spatialdata.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_core/spatialdata.py:1192)
- [io_table.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_io/io_table.py:50)

Observed locally for `sdata.write(...)`:

- Dask-backed `.X` was written to `tables/<name>/X`
- Dask-backed `obsm["feat"]` was written to `tables/<name>/obsm/feat`
- the on-disk arrays were Zarr arrays
- the in-memory `SpatialData.tables[...]` object still had Dask arrays after write

Source inspection strongly suggests `write_element(...)` has the same serialization behavior for tables, since it reaches the same `write_table(...)` implementation.

### Reading with `spatialdata.read_zarr(...)`

Observed locally:

- tables were materialized as normal in-memory `AnnData`
- Dask-backed dense `.X` and dense `obsm[...]` came back as eager arrays

So the standard SpatialData read path does not preserve lazy table matrices.

### Reading a SpatialData table lazily

Observed locally:

- if we bypass `spatialdata.read_zarr(...)`
- and point `anndata.experimental.read_lazy(...)` directly at the table group inside the SpatialData Zarr store

then:

- `.X` can be reopened lazily
- `obsm[...]` can be reopened lazily

In other words:

- SpatialData writes table data into a form that is compatible with lazy AnnData reading
- but SpatialData itself does not currently expose lazy table loading through its normal API

## Practical Takeaways for `napari-harpy`

### Feature storage

Using:

- `obsm[feature_key]` for the array-like feature matrix
- `uns["harpy_feature_matrices"][feature_key]` for companion metadata

remains a good design.

Why:

- it matches what `AnnData` can store today
- it leaves a path open for Dask-backed feature matrices in memory
- it does not force the feature extractor to depend on DataFrame semantics

### Write semantics

We can write Dask-backed tables to a SpatialData store without losing laziness in the current in-memory object.

This is good for:

- feature extraction workflows that update in-memory tables
- explicit write-to-zarr operations afterward

### Read semantics

If we reopen the dataset with:

- `spatialdata.read_zarr(...)`

then tables will be eager in memory.

If we need lazy table matrices after reopening, we would need a different table-loading path, for example:

- open the SpatialData store normally for spatial elements
- open selected table groups lazily with `anndata.experimental.read_lazy(...)`

### Important caveat

Even if `.X` or `obsm[...]` is lazy, downstream code may still realize it immediately.

Examples:

- `np.asarray(...)`
- scikit-learn estimators that require eager arrays
- code that assumes dense NumPy semantics

So lazy storage and lazy reopening do not automatically imply end-to-end lazy computation.

## Suggested Contract

For feature matrices specifically, it is safest to assume:

- supported in-memory backends: `np.ndarray`, `scipy.sparse.spmatrix`, `dask.array.Array`
- supported persisted representation: Zarr-backed arrays written through AnnData / SpatialData
- supported lazy reopening path: `anndata.experimental.read_lazy(...)`
- unsupported expectation: that `spatialdata.read_zarr(...)` will preserve lazy tables

## References

Official docs:

- AnnData Dask support tutorial: https://anndata.readthedocs.io/en/stable/tutorials/notebooks/anndata_dask_array.html
- AnnData `read_lazy`: https://anndata.readthedocs.io/en/stable/generated/anndata.experimental.read_lazy.html
- AnnData API: https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html
- SpatialData API: https://spatialdata.scverse.org/en/latest/api/SpatialData.html
- SpatialData models: https://spatialdata.scverse.org/en/stable/api/models.html

Local source inspected:

- [anndata/_core/aligned_mapping.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/anndata/_core/aligned_mapping.py:70)
- [anndata/_core/anndata.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/anndata/_core/anndata.py:1831)
- [spatialdata/models/models.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/models/models.py:1073)
- [spatialdata/_core/_elements.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_core/_elements.py:121)
- [spatialdata/_core/spatialdata.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_core/spatialdata.py:1192)
- [spatialdata/_io/io_table.py](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/spatialdata/_io/io_table.py:50)
