Investigated, no code changed.

**Short Answer**
I would implement pixel classification as a new, separate pipeline beside the existing object-classification pipeline.

For the first feature extractor, I would choose a **small convolutional feature extractor**, not DINO first. My pick would be:

**ConvNeXt-Tiny early-layer features**, with raw normalized channel intensities appended, and with optional PCA/random-projection compression before caching.

Reason: for MACSima / PhenoCycler-style multiplex data, we care a lot about local marker intensity, membrane/nuclear texture, boundaries, and neighborhood context across many channels. A ViT/foundation model like DINOv2 is attractive, but it is RGB-oriented, patch-based, heavier, and awkward for 20-50 fluorescence channels. Convpaint itself recommends VGG/Ilastik-like local features for “what does it look like here?” tasks, and DINO/JAFAR for contextual/semantic tasks. For multiplex pixel classification, I think local, dense, tile-friendly CNN features are the better first bet.

**What napari-harpy Already Has**
napari-harpy currently has:

- object-level feature extraction via `harpy.tb.add_feature_matrix(...)`, writing into `AnnData.obsm`
  - [feature_extraction/controller.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/feature_extraction/controller.py:191)
- object classification using `RandomForestClassifier`, reading `.obsm[feature_key]`, then writing `pred_class` / `pred_confidence`
  - [object_classification/controller.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/object_classification/controller.py:186)
- a shared `HarpyAppState` event pattern for cross-widget updates
  - [\_app_state.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_app_state.py:86)

Pixel classification should not be forced into `.obsm`, because dense per-pixel features are image-like, not table-row-like.

**What Convpaint Does**
Convpaint’s core pattern is:

- `ConvpaintModel` combines feature extractor + classifier + parameters.
- Feature extraction normalizes, pads, tiles, handles channel selection, builds feature pyramids, and concatenates features.
- Non-RGB multichannel data is handled by repeating each channel to the model’s expected input channel count and extracting per-channel features.
- Training extracts only annotated-pixel features.
- Prediction flattens feature images, runs `predict_proba`, and restores image shape.
- “memory_mode” caches annotated-pixel feature rows in a model-local DataFrame, not a persistent dense feature image cache.

Useful source anchors:
- Convpaint model API and training/prediction overview: https://github.com/guiwitz/napari-convpaint/blob/b1e1dddeca12fd363fe340a81292f6656cc20a90/src/napari_convpaint/convpaint_model.py#L764-L915
- Convpaint feature pipeline: https://github.com/guiwitz/napari-convpaint/blob/b1e1dddeca12fd363fe340a81292f6656cc20a90/src/napari_convpaint/convpaint_model.py#L919-L1131
- multichannel handling: https://github.com/guiwitz/napari-convpaint/blob/b1e1dddeca12fd363fe340a81292f6656cc20a90/src/napari_convpaint/feature_extractor.py#L441-L504
- classifier path: https://github.com/guiwitz/napari-convpaint/blob/b1e1dddeca12fd363fe340a81292f6656cc20a90/src/napari_convpaint/convpaint_model.py#L1189-L1278
- cache-like memory mode: https://github.com/guiwitz/napari-convpaint/blob/b1e1dddeca12fd363fe340a81292f6656cc20a90/src/napari_convpaint/convpaint_model.py#L1355-L1505

**Recommended Design**
Add a new `PixelClassificationWidget` and controller.

Pipeline:

1. User selects SpatialData image, coordinate system, and channels.
2. User paints scribble labels in a napari `Labels` layer: `0 = unlabeled`, `1..N = classes`.
3. Feature extractor produces tile-wise deep feature arrays, conceptually `(F_raw, y, x)`.
4. Build a pixel feature cache by concatenating normalized marker intensities with reduced deep features.
5. Train classifier from annotated pixel coordinates only.
6. Predict labels/probabilities tile-wise from cached features.
7. Write output as a new labels element, or initially as a viewer layer with optional “save to SpatialData”.

**Pixel Feature Cache Schema**
The cached pixel feature stack should make the marker-intensity and deep-feature parts explicit:

```text
cached_features = concat(
    normalized_selected_marker_intensities,  # C planes
    reduced_deep_features                    # F_reduced planes
)

shape: (C + F_reduced, y, x)
```

Here `C` is the number of selected image channels. These planes are the normalized per-pixel marker
intensities from the source multiplex image, for example DAPI, CD3, CD20, PanCK, and so on. They are
not deep features. They should be normalized consistently before caching, for example with percentile
clipping plus scaling, and optionally log/asinh transformation if that becomes part of the selected
preprocessing profile.

`F_raw` is the number of raw deep feature planes produced by the feature extractor before reduction.
For multiplex data this can grow very quickly, for example:

```text
30 selected channels x 64 CNN feature planes x 3 scales = 5760 raw deep feature planes
```

The reduction step applies only to this deep-feature axis:

```text
(F_raw, y, x) -> (F_reduced, y, x)
```

The spatial resolution stays unchanged. A practical implementation can sample pixels from the raw
deep-feature cache stream, fit a reducer such as sampled PCA, `IncrementalPCA`, or a fixed random
projection, and then transform every tile before writing it to the final cache. The original marker
intensity planes should be appended back after reducing the deep features, because rare marker
signals can be biologically important and should not be washed out by PCA.

Example:

```text
C = 30 selected marker channels
F_reduced = 64 deep-feature components
cached feature stack = (94, y, x)
```

The reducer is part of the feature schema. A classifier trained on this cache is valid only for the
same source image, selected channels, normalization, feature extractor, model weights, layer/scaling
settings, and reducer parameters.

**Back-of-the-envelope Cache Scaling**
Pixel classification will usually run on a curated channel subset, not necessarily all multiplex
channels. For MACSima / PhenoCycler-style workflows, a typical first target is probably `1..5`
selected marker channels.

This matters twice:

1. the temporary raw deep feature stream is smaller before reduction;
2. the persistent cache has fewer marker-intensity planes.

For a VGG-like early-layer extractor with `64` feature planes per selected channel and `3` scales:

```text
5 selected channels  x 64 features x 3 scales = 960 raw deep feature planes
30 selected channels x 64 features x 3 scales = 5760 raw deep feature planes
```

So channel selection is the first scaling control. The reducer is the second.

If the cache stores `C` normalized marker-intensity planes plus `F_reduced` reduced deep-feature
planes, the persistent size is approximately:

```text
(C + F_reduced) * y * x * bytes_per_value
```

Assuming `float16` or `uint16`-like storage (`2 bytes/value`), common examples are:

| selected marker channels | reduced deep features | total planes |
| ---: | ---: | ---: |
| 1 | 64 | 65 |
| 5 | 64 | 69 |
| 5 | 32 | 37 |
| 30 | 64 | 94 |

Approximate persistent cache sizes:

| image size | 37 planes | 65 planes | 69 planes | 94 planes |
| ---: | ---: | ---: | ---: | ---: |
| `2048 x 2048` | `0.29 GiB` | `0.51 GiB` | `0.54 GiB` | `0.73 GiB` |
| `4096 x 4096` | `1.16 GiB` | `2.03 GiB` | `2.16 GiB` | `2.94 GiB` |
| `8192 x 8192` | `4.63 GiB` | `8.13 GiB` | `8.63 GiB` | `11.75 GiB` |
| `10000 x 10000` | `6.89 GiB` | `12.11 GiB` | `12.85 GiB` | `17.51 GiB` |
| `20000 x 20000` | `27.57 GiB` | `48.43 GiB` | `51.41 GiB` | `70.04 GiB` |

The practical MVP default should probably be conservative, for example `C + 16` or `C + 32` planes,
with `C` usually between `1` and `5`. A `5 + 32 = 37` plane cache is much more manageable than a
`30 + 64 = 94` plane cache, while still giving the classifier texture/context features beyond raw
marker intensity alone.

**Cache Recommendation**
Do not cache raw high-dimensional deep features blindly. For multiplex data this explodes quickly.

Better:

- cache tile-wise normalized marker intensities plus reduced deep features, e.g. `float16`, shape
  `(C + F_reduced, y, x)`
- preserve the selected marker-intensity planes separately from the deep feature reducer
- use a manifest keyed by:
  - source image element
  - coordinate system
  - selected channels
  - model name/version/weights
  - preprocessing/normalization
  - layer/scaling settings
  - PCA/projection settings
  - output shape, axes, dtype, cache schema version

Convpaint’s memory mode is useful inspiration for avoiding repeated annotation-feature extraction, but napari-harpy should use a persistent feature-image cache because whole-image prediction and re-use across sessions matter here.

**Model Choice**
My first choice: **ConvNeXt-Tiny early convolutional layers**, tile-friendly, modern, finite receptive field, and faster than transformer/foundation options.

Fallback conservative baseline: **VGG16 first layer + scales `[1, 2, 4]`**, because Convpaint’s docs explicitly recommend early VGG layers for local texture/color/edge tasks and its source already validates that design.

I would not start with DINOv2/JAFAR for multiplex. Convpaint documents DINOv2/JAFAR as strong for semantic/contextual tasks and high spatial resolution, but DINO-style RGB patch features are costly and awkward for many-channel fluorescence. Good future optional backend, not MVP.

Sources:
- Convpaint feature extractor guidance: https://guiwitz.github.io/napari-convpaint/book/FE_descriptions.html
- Convpaint parameters/cache settings: https://guiwitz.github.io/napari-convpaint/book/Params_settings.html
- DINOv2 official repo/model notes: https://github.com/facebookresearch/dinov2
