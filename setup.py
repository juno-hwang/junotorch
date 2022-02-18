import setuptools

setuptools.setup(
    name="junotorch", # Replace with your own username
    version="0.3.1",
    author="Juno Hwang",
    author_email="wnsdh10@snu.ac.kr",
    description="Custom pytorch modules that I use often",
    url="https://github.com/juno-hwang/junotorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','matplotlib','torch'],
    python_requires='>=3.6',
)