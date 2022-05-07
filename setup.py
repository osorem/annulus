from struct import pack
from setuptools import setup


setup(
    name='anulus',
    version='0.0.3beta',
    author = "OJ",
    description = "Intredit traffic sign detection and classifier",
    url = "https://github.com/osorem/anulus",
    packages=['anulus'],
    install_requires=[
        'opencv-python',
        'numpy',
        'pydantic',
        'scikit-image',
        'pillow',
    ],
    package_dir={'anulus': 'src/anulus'},
    package_data = {"anulus": ["data/*.png", "matchers/*.png"]}
)
