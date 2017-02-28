from setuptools import setup

from pmttools import __version__

setup(name='pmttools',
      version=__version__,
      description='Analysis of photo multiplier response',
      long_description='Photo multiplier tubes (PMTs) are used widely in high energy physics applications. The here provided tools shall help with their characterization in the lab.',
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/pmttolls',
      #download_url="pip install pyosci",
      install_requires=['numpy>=1.11.0',
                        'matplotlib>=1.5.0',
                        'appdirs>=1.4.0',
                        'pyprind>=2.9.6',
                        'six>=1.1.0'],
      license="GPL",
      platforms=["Ubuntu 14.04","Ubuntu 16.04"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Physics"
              ],
      keywords=["PMT", "photo multiplier tubes",\
                "HEP",\
                "physics", "engineering", "callibration", "characterization"],
      packages=['pmttools'],
      #scripts=[],
      #package_data={'pyosci': ['pyoscidefault.mplstyle','pyoscipresent.mplstyle']}
      )
