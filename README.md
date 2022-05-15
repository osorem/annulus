# Annulus detection and classification
## Quick Test

Test here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ItCdWchvthUPo5SaxndawkshjTbSjXqd?usp=sharing)


## What is this?

This is a Python package that uses color thresholding and other classical vision methods through OpenCV to detect a subset of prohibition traffic signs. It will detect and classify any sign that is red.


## How to Use?

1. This package is not hosted on PyPi so don't try installing it with pip like that. Just do this:

```bash
python3.10 virtualenv venv
source venv/bin/activate (on Linux) or venv/vin/activate.ps1 (Windows)
python3.10 pip install git+https://github.com/osorem/annulus.git
```

2. Create a new Python file and:

```python
from annulus import Settings, annulus_detect
import cv2
from pprint import pprint

st = Settings()

det, ssim_scores, coords, color_isolated = annulus_detect("/path/to/img.png", st)

pprint(ssim_scores)
pprint(coords)


cv2.imshow('Detected Signs', det)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
An exhaustive list of settings + grid search to find the best params using `grid_search.py` will be added to this readme pretty soon. Meanwhile please use the `help(Settings)`.
