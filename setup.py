from setuptools import setup, find_packages
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(ROOT_DIR, 'README.md')

setup(
    name='pascalvoc-ap',
    author='devbruce',
    author_email='bruce93k@gmail.com',
    description='PascalVOC AP with Python',
    long_description=open(README_PATH, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    version='1.0.2',
    url='https://github.com/DevBruce/PascalVOC-AP-Python',
    packages=find_packages(),
    python_requires='>=3.7',
    keywords=[
        'pascalvoc',
        'object_detection',
        'metric',
        'evaluation',
        'ap',
        'python',
        ],
)
