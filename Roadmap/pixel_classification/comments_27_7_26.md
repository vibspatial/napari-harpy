Yes—we are ready to start implementing pixel classification, specifically Slice 1. The product direction and end-to-end workflow are sufficiently settled.

I would not yet describe every part of Slices 1–6 as completely frozen. A short final clarification pass is needed before reaching the affected slices, but none requires reconsidering the overall design.

## Readiness by slice

| Slice | Readiness | Remaining work |
|---|---|---|
| 1. Scale and transformations | Ready | Implement with alignment tests first |
| 2. Widget and target selection | Ready | Mostly reuse existing discovery/UI patterns |
| 3. Annotation lifecycle | Almost ready | Resolve shared coordinate-change dirty guards |
| 4. Training | Almost ready | Define immutable training snapshot and classifier-ID creation |
| 5. Prediction | Almost ready | Define a concrete prediction-block memory budget |
| 6. Persistence | Needs a technical mini-spec | Finalize manifest serialization and Harpy write mechanics |
| 7–9 | Roadmap-ready | Detailed APIs can wait until the first milestone is stable |

## Remaining clarifications

### 1. Shared coordinate-system dirty guards

The roadmap requires unsaved annotation protection when changing coordinate systems or targets [in Slice 3](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1377).

Currently, `HarpyAppState` permits only one coordinate-system change participant [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_app_state.py:371), and the Annotation widget already occupies that role [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/annotation/widget.py:195).

Pixel Classification therefore cannot independently register its dirty guard. Before Slice 3, I recommend extending this into a shared multi-participant preflight: every active editor gets the opportunity to accept or cancel a coordinate-system change.

This is the only repository-architecture issue I consider a genuine prerequisite for the editable workflow.

### 2. Immutable training-job snapshot

Slice 4 says training runs in a worker and returns an immutable result tied to revisions [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1423), but it does not say what happens if the user paints while the worker scans the annotation array.

The worker must not read an array that napari may mutate concurrently.

My recommendation:

- capture a private `uint8` annotation snapshot, class schema, channels, target and revisions when `Train` is pressed;
- allow the user to continue editing;
- accept the result only if the captured revisions remain current;
- otherwise discard it as obsolete.

This follows the existing object-classification pattern of passing a prepared immutable job to the worker [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/object_classification/controller.py:221).

### 3. Create `classifier_id` during Slice 4

Prediction provenance already requires a classifier ID [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1146), but the Slice 4 training-result specification does not explicitly include one [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1437).

The ID should be created for every successful training result in Slice 4. Slice 7 should serialize that existing ID into the bundle, not create a new identity during export.

### 4. Concrete prediction block limit

Slice 5 requires bounded blocks [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1469), but does not define the bound. Following source Zarr chunks alone is insufficient because an image can have unusually large chunks.

Before Slice 5, we should choose a versioned byte budget for:

```text
source block
+ float32 pixels × channels matrix
+ finite mask
+ temporary predictions
```

The tile planner should subdivide source chunks when necessary. This can remain an internal product constant rather than a UI control.

### 5. Persistence mechanics need one technical mini-spec

The persistence behavior is thoroughly specified from the user’s perspective, but Slice 6 still needs:

- the complete annotation-mode JSON example and required/optional field rules;
- whether the `.harpy-cache.zarr` sidecar is treated as a Zarr hierarchy, a filesystem project directory, or a defined mixture;
- temporary-manifest naming and atomic finalization;
- how an explicitly selected non-default sidecar is reselected in a later session.

There is also one likely redundancy: the roadmap says to call `harpy.im.add_labels(...)` and provide separate `SpatialData.write_element(...)` support [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1541). The installed Harpy implementation already writes new backed elements and performs on-disk replacement for overwrites [here](/Users/arne.defauw/VIB/napari_harpy/.venv/lib/python3.13/site-packages/harpy/image/_manager.py:202). We should clarify that there is no second `write_element()` call after `add_labels()`.

### 6. A few Slice 3 acceptance criteria belong to later slices

Slice 3 currently includes acceptance criteria for:

- persisted class-schema reload;
- classifier/prediction staleness;
- rendering retained stale predictions.

Those features do not exist until Slices 4–6 [here](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_17_7_26.md:1395). The behavior itself is correct, but those tests should be moved to the slices that introduce the required components.

## Repository fit

The planned implementation fits the current architecture well:

- actual multiscale keys are already enumerated in the Histogram widget [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:709);
- exact scale resolution already has a Qt-free precedent [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/histogram.py:130);
- the viewer adapter already converts SpatialData transformations into napari affines [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/adapter.py:2472);
- SpatialData naming validation is reusable [here](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/validation.py:20);
- worker, controller, persistence and classifier-export patterns already exist.

The selected-level-to-image transform is new and should remain the focus of Slice 1 rather than being buried inside the widget.

My conclusion: start Slice 1 now. Before beginning Slice 3, settle the shared dirty-guard architecture; before Slice 4, settle training snapshots and classifier IDs; before Slices 5 and 6, write the small block-budget and persistence mini-specs. After those clarifications, the complete first milestone will be implementation-ready.