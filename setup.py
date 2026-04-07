from setuptools import setup, find_packages

setup(
    name="HyperCrystal",
    version="2.0.0",
    description="Quantum-Inspired Novelty Engine",
    author="HyperCrystal Contributors",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=3.0.0",
        "flask-socketio>=5.3.0",
        "flask-limiter>=3.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "cma>=3.2.0",
        "tqdm>=4.65.0",
        "pyjwt>=2.8.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "HyperCrystal=HyperCrystal.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
