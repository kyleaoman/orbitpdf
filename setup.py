from setuptools import setup

setup(
    name="orbitpdf",
    version="0.1",
    description="Calculation of orbital libraries and pdfs from simulations.",
    url="",
    author="Kyle Oman",
    author_email="kyle.a.oman@durham.ac.uk",
    license="",
    packages=["orbitpdf"],
    install_requires=["numpy", "astropy", "h5py", "pathos", "read_tree"],
    include_package_data=True,
    zip_safe=False,
)
