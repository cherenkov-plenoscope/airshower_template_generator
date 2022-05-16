import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="airshower_template_generator",
    version="0.1.1",
    description="Generate Cherenkov-light-templates of cosmic-ray airshowers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cherenkov-plenoscope/airshower_template_generator.git",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=["airshower_template_generator",],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    install_requires=[
        "json_numpy_relleums",
        "binning_utils_relleums",
        "corsika_primary",
        "sebastians_matplotlib_addons",
        "plenoirf",
        "queue_map_reduce",
        "scipy",
    ],
)
