"""
Fake Review Archaeology - Setup Configuration
=============================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

setup(
    name='fake-review-archaeology',
    version='1.0.0',
    author='Data Analytics Team',
    author_email='data-analytics@company.com',
    description='AI-Powered Detection of Synthetic and Fraudulent Product Reviews',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/fake-review-archaeology',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
            'pre-commit>=3.3.0',
        ],
        'gpu': [
            'torch>=2.0.0+cu118',
        ],
    },
    entry_points={
        'console_scripts': [
            'fake-review-train=train:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
