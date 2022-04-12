from distutils.core import setup

setup(
    name="cpcn",
    version="0.0.1",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    url="https://github.com/ttesileanu/cpcn",
    packages=["cpcn"],
    install_requires=[
        "numpy",
        "scipy",
        "setuptools",
        "torch",
        "torchvision",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pydove",
        "ipykernel",
    ],
)
