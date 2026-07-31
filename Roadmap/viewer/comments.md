Overall, Slice 3b is architecturally sound. I don’t see a reason to unwind the shared-row approach, and the important feedback-loop and stale-binding cases are well protected.

The current ownership model is especially clear:

- The shared row presents napari state and emits user intent, but never mutates napari itself: [_OverlayChannelRow](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/overlay_channel_row.py:74).
- It listens directly to `layer.events.visible` and `layer.events.colormap`: [_connect_presentation_events](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/overlay_channel_row.py:184).
- Histogram revalidates the exact live binding before acting: [_resolve_exact_card_overlay_binding](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1153).
- Rows retain a fixed binding and are reconstructed when it changes, both in the [Viewer](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/viewer/image_widget.py:261) and [Histogram](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1083).

That separation is worth preserving.

### Cleanup I recommend

1. Decouple overlay resolution from Histogram scale validation

This is the most substantive issue I found.

Histogram overlay matching calls [_resolve_card_target](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1471), which rejects the target when Histogram scale resolution failed at [line 1483](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1483).

Consequently, a scale problem can:

- Disable loading an otherwise valid overlay.
- Remove an existing shared overlay row from the Histogram UI.
- Prevent visibility and colormap controls from working.

Overlay operations only need coordinate system, image, and channel. They should not depend on whether a valid Histogram calculation scale exists.

I recommend extracting a small overlay-target resolver and leaving scale validation in the Histogram calculation path.

2. Improve viewer-action feedback state

`viewer_action_message` is only a string: [_HistogramCard](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:196). It gets appended to whatever Histogram status style is currently active: [_update_card_status](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:971).

This creates two small inconsistencies:

- “Overlay loaded in viewer” can remain after the layer is removed directly in napari because membership reconciliation does not update that message: [_on_image_overlay_layers_changed](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1038).
- An overlay mutation error may be rendered inside a success-styled Histogram status card.

A small `message + kind` viewer-feedback state would solve this without introducing a larger status framework.

3. Avoid displaying a stale swatch for unsupported colormaps

If napari’s colormap cannot be converted into a supported presentation, [_sync_colormap_from_layer](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/overlay_channel_row.py:212) simply returns.

That leaves the previous solid color or gradient displayed, even though it no longer represents the layer. A neutral “unsupported colormap” presentation would be more truthful.

4. Add whole-widget teardown for Histogram subscriptions

Removing an individual card correctly disconnects its row and contrast callbacks: [remove_histogram_card](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:319).

The contrast-limit subscription, however, uses a lambda capturing the widget: [_refresh_card_contrast_sync](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1286). It is disconnected during card cleanup at [_disconnect_card_contrast_sync](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/histogram/widget.py:1356), but there is no explicit whole-widget teardown.

This is low risk during normal application lifetime, but an explicit `dispose()` covering all cards would make ownership complete and easier to test.

### What I would leave alone

I would not centralize the Viewer and Histogram mutation handlers. They look similar, but each owner has different contextual feedback and reconciliation responsibilities. Sharing the presentation row while keeping mutations owner-local is the right boundary.

I also would not make the shared row replace its binding in place. Reconstructing it when the layer changes makes stale napari subscriptions much easier to reason about.

The existing tests already cover the most dangerous behavior:

- Rows emit intent without mutating napari: [test](/Users/arne.defauw/VIB/napari_harpy/tests/test_overlay_channel_row.py:27).
- Disposal stops native presentation updates: [test](/Users/arne.defauw/VIB/napari_harpy/tests/test_overlay_channel_row.py:52).
- Stale rows cannot mutate a newly selected target: [test](/Users/arne.defauw/VIB/napari_harpy/tests/test_histogram_widget.py:792).
- Bidirectional visibility and colormap synchronization works: [test](/Users/arne.defauw/VIB/napari_harpy/tests/test_histogram_widget.py:841).
- Multiple Histogram rows can safely observe the same layer: [test](/Users/arne.defauw/VIB/napari_harpy/tests/test_histogram_widget.py:999).

My suggested cleanup order would therefore be: overlay-target decoupling first, feedback consistency second, and the colormap/teardown hardening afterward. No urgent redesign is needed.