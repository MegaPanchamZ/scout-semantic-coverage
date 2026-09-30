"""Stub for ``imgaug.augmenters`` (see the parent package docstring)."""

_MESSAGE = (
    "imgaug.augmenters is stubbed in this environment because imgaug 0.4.0 is "
    "incompatible with NumPy 2.x. Image augmentation is a training-only path."
)


def __getattr__(name: str):
    raise NotImplementedError(f"{_MESSAGE} (requested: imgaug.augmenters.{name})")
