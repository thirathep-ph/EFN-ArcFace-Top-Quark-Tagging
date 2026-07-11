from setuptools import setup, find_packages

setup(
    name="arcjet",
    version="0.1.0",
    python_requires=">=3.10, <3.13",
    packages=find_packages(),
    package_data={'arcjet': ['config.json']},
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
        "umap-learn",
    ],
    entry_points={
        'console_scripts': [
            'arcjet-train=scripts.train_arcface:main',
            'arcjet-evaluate=scripts.evaluate_model:main',
            'arcjet-baselines=scripts.train_baselines:main',
        ],
    },
    author="Thirathep Naowabut Phiankham",
    description="ArcJet: Ante-hoc Interpretable Jet Tagging with EFN and ArcFace Hyperspherical Prototypes",
)
