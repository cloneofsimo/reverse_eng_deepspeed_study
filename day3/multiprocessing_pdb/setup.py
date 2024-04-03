from setuptools import setup, find_packages

setup(
    name="multiprocessing_pdb",
    version="0.1.0",
    packages=find_packages(),
    description="A multiprocessing-compatible Python debugger.",
    author="Seunghyun Seo",
    author_email="real.seunghyun.seo@navercorp.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)