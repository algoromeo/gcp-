from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ['Keras==2.1.6' ,'h5py==2.7.0', 'tensorflow-gpu==1.5', 'SoundFile==0.10.2', 'librosa==0.6.2']

setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      description='Gender Classification',
      author='Bhargav Desai',
      author_email='desaibhargav98@gmail.com',
      include_package_data=True,
      license='MIT'
)  