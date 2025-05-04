from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

APP_NAME = 'PyMemLab'
VERSION = '0.1.0'
AUTHOR = 'Sina Gilassi'
EMAIL = "<sina.gilassi@gmail.com>"
DESCRIPTION = 'A Python toolkit for modeling, simulation, and optimization of membrane-based gas separation processes.'
LONG_DESCRIPTION = "PyMemLab a versatile and extensible Python package designed for engineers, researchers, and scientists working in the field of membrane separations. It provides a laboratory-like environment where users can model membrane transport, simulate separation processes, and perform parametric studies or process optimization. Whether you're designing new membrane modules, evaluating separation performance, or testing novel materials, **PyMemLab** offers the computational tools to support your work."

# Setting up
setup(
    name=APP_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    include_package_data=True,  # Make sure to include non-Python files
    # Add both config and data files
    package_data={'': ['config/*.yml', 'config/*.csv', 'data/*.csv',
                       'data/*.yml', 'plugin/*.yml', 'plugin/*.csv']},
    license='MIT',
    license_files=[],
    install_requires=['pandas', 'numpy',
                      'PyYAML', 'PyCUC', 'scipy'],
    extras_require={
        "plotting": ["matplotlib"],
    },
    keywords=[
        'python', 'chemical-engineering', 'process-modeling',
        'membrane-separation', 'gas-separation', 'membrane-transport',
        'process-simulation', 'process-optimization', 'thermodynamics',
        'membrane-technology', 'separation-processes', 'membrane-modules',
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.13",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.10',
)
