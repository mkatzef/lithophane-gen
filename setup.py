from setuptools import setup

DEPENDENCIES = [
	'matplotlib',
	'numpy',
	'numpy-stl',
	'imageio',
	'Pillow'
]

setup(
   name='lithogen',
   version='0.1',
   description='Lithophane generator',
   author='Marc Katzef',
   author_email='marckatzef@gmail.com',
   packages=['lithogen'],
   install_requires=DEPENDENCIES #external packages as dependencies
)
