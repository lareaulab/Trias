from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="trias",
    version="0.1",
    description="A generative language model for codon sequence design",
    author="Marjan Faizi",
    license="MIT",
    python_requires=">=3.10,<3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
)