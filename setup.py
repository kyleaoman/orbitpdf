from setuptools import setup, Extension

setup(
    name='orbitpdf',
    version='0.1',
    description='''Calculation of orbital libraries and pdfs from simulations.''',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='',
    packages=['orbitpdf'],
    install_requires=['numpy', 'astropy', 'h5py'],
    include_package_data=True,
    zip_safe=False
)

rt_base = 'read_tree/'
read_tree_module = Extension(
    '_read_tree',
    sources = [rt_base + f for f in (
        'read_tree_wrap.c',
        'read_tree.c',
        'check_syscalls.c',
        'stringparse.c',
        'strtonum.c'
    )],
    define_macros = [('SWIG', None)]
)

setup(
    name = 'read_tree',
    version = '1.0',
    author = 'Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='',
    description = '''Wrapper for trees read utility.''',
    ext_modules = [read_tree_module],
    py_modules = ['read_tree'],
    include_package_data=True,
    zip_safe=False
)
