from setuptools import setup, find_packages

setup(
    name="jet_tagging",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "awkward",
        "vector",
        "h5py",
        "hdf5plugin",
        "fastjet",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    author="Your Name",
    description="Interpretable Jet Tagging with EFN and ArcFace Prototypes",
)
