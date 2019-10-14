from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='FRET-board',
    version='0.0.2',
    packages=['FRETboard'],
    install_requires=['numpy==1.17.2',
                      'pandas==0.25.1',
                      'pomegranate==0.11.1',
                      'bokeh==1.3.4',
                      'cached-property==1.5.1',
                      'tabulate==0.8.3'],
    author='Carlos de Lannoy',
    author_email='carlos.delannoy@wur.nl',
    description='Supervise FRET event detection algorithms',
    long_description=readme(),
    license='GPL-3.0',
    keywords='FRET Forster resonance energy transfer supervised machine learning',
    url='https://github.com/cvdelannoy/FRETboard',
    entry_points={
        'console_scripts': [
            'FRETboard = FRETboard.__main__:main'
        ]
    },
    include_package_data=True
)
