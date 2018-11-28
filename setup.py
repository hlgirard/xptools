from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='xptools',
      version='0.37post1',
      description='Image and video analysis tools for experimental sciences',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/hlgirard/xptools',
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
          'tqdm'
      ],
      scripts=[
          'bin/analyze_front',
          'bin/display_image_matrix',
          'bin/analyze_bubbles'
      ],
      zip_safe=False)