from setuptools import setup

from pmttools import __version__

def parse_requirements(req_file):
    with open(req_file) as f:
        reqs = []
        for r in f.readlines():
            if not r.startswith("http"):
                reqs.append(r)

        return reqs

try:
    requirements = parse_requirements("requirements.txt")
except Exception as e:
    print ("Failed parsing requiremnts, installing dummy requirements...")
    requirements = ['numpy>=1.9.0',
                     'matplotlib>=1.5.0',
                     'pyevsel>=0.0.6',
                     'futures>=3.0.5',
                     'future>=0.16.0',
                     'pyprind>=2.9.6']



setup(name='pmttools',
      version=__version__,
      description='Analysis of photo multiplier response',
      long_description='Photo multiplier tubes (PMTs) are used widely in high energy physics applications. The here provided tools shall help with their characterization in the lab.',
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/pmttolls',
      #download_url="pip install pyosci",
      install_requires=requirements,
      setup_requires=["pytest-runner"],
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
      tests_require=["pytest"],
      keywords=["PMT", "photo multiplier tubes",\
                "HEP",\
                "physics", "engineering", "callibration", "characterization"],
      packages=['pmttools'],
      #scripts=[],
      #package_data={'pyosci': ['pyoscidefault.mplstyle','pyoscipresent.mplstyle']}
      )
