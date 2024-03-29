import setuptools
import os


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("airshower_template_generator", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


setuptools.setup(
    name="airshower_template_generator_cherenkov-plenoscope",
    version=version,
    description="Generate Cherenkov-light-templates of cosmic-ray airshowers.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/airshower_template_generator",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "airshower_template_generator",
    ],
    package_data={"airshower_template_generator": []},
    install_requires=[
        "json_utils_sebastian-achim-mueller",
        "binning_utils_sebastian-achim-mueller",
        "spherical_coordinates>=0.1.1",
        "corsika_primary>=2.4.4",
        "atmospheric_cherenkov_response_cherenkov-plenoscope-project",
        "sebastians_matplotlib_addons",
        "rename_after_writing",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
