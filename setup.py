from setuptools import setup, find_packages
setup(name='xpanalysis',
      version='0.1',
      description='Image and video analysis tools for experimental sciences',
      url='https://github.com/hlgirard/ImageAnalysis',
      author='Henri-Louis Girard',
      author_email='hl.girard@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scikit-image',
          'joblib',
          'matplotlib',
          'av',
      ],
      scripts=['bin/analyze_front'],
      zip_safe=False)