"""Inference-time stub for the optional ``imgaug`` training dependency.

The PCLA InterFuser agent vendors code that imports ``imgaug`` at module load
time (``timm/data/augmenter.py``), but augmentation is only instantiated when
``augment_prob > 0``. Inference runs use ``augment_prob = 0.0``, so the real
library is never exercised.

The genuine ``imgaug`` 0.4.0 release is incompatible with NumPy 2.x
(``np.sctypes`` was removed), and this environment runs NumPy 2.x. This stub
keeps the import chain intact and fails with an explicit message if any code
path actually tries to use augmentation.
"""

__version__ = "0.4.0-scout-stub"

_MESSAGE = (
    "imgaug is stubbed in this environment because imgaug 0.4.0 is incompatible "
    "with NumPy 2.x. Image augmentation is a training-only path; if you truly "
    "need it, install imgaug in a NumPy 1.x environment."
)

from . import augmenters  # noqa: E402,F401  (real submodule; its symbols raise on use)


def __getattr__(name: str):
    raise NotImplementedError(f"{_MESSAGE} (requested: imgaug.{name})")
