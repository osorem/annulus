from struct import pack
from setuptools import setup


setup(
    name='annulus',
    version='0.0.3beta',
    author = "OJ",
    description = "Intredit traffic sign detection and classifier",
    url = "https://github.com/osorem/annulus",
    packages=['annulus'],
    install_requires=[
        'opencv-python',
        'numpy',
        'pydantic',
        'scikit-image',
        'pillow',
    ],
    package_dir={'annulus': 'src/annulus'},
    package_data = {"annulus": ["data/*.png", "matchers/*.png"]}
)
