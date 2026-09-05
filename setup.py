from setuptools import setup
from setuptools import find_packages

# change this.
NAME = "abundanance_aware_mgm"
AUTHOR = "Lewis Msasa"
EMAIL = "lmsa327@aucklanduni.ac.nz"
URL = ""
LICENSE = "MIT"
DESCRIPTION = "Adopted from MGM (Microbial General Model) by HUST-NingKang-Lab as a large-scaled pretrained language model for interpretable microbiome data analysis adding abundance awareness."


if __name__ == "__main__":
    setup(
        name=NAME,
        version="0.0.1",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        license=LICENSE,
        description=DESCRIPTION,
        packages=find_packages(),
        package_dir={'abundance_aware': 'abundance_aware'},
        include_package_data=True,
        install_requires=open("./requirements.txt", "r").read().splitlines(),
        long_description=open("./README.md", "r").read(),
        long_description_content_type='text/markdown',
        # change package_name to your package name.
        entry_points={
            "console_scripts": [
                "gwmm=abundance_aware.cli:main"
            ]
        },
        package_data={
            # change package_name to your package name.
            "config": ["./resources/config.ini"],
            "general_model": ["./resources/general_model"],
			"phylo":["./resources/phylogeny.csv"],
            "MicrobialTokenizer":["./resources/MicrobialTokenizer.pkl"],
			"tmp":["./resources/tmp"]
        },
        zip_safe=True,
        classifiers=[
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Programming Language :: Python :: 3.14",
            "Development Status :: 4 - Beta",
            "Operating System :: OS Independent",
            "Natural Language :: English"

        ],
        python_requires='>=3.10',
    )