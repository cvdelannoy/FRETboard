from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='FRET-board',
    version='0.0.4',
    packages=['FRETboard'],
    install_requires=['joblib==0.14.1',
                      'cython==0.29.20',
                      'pandas==1.5.2',
                      'statsmodels==0.13.2',
                      'pomegranate==0.14.9',
                      'bokeh==2.4.2',
                      'cached-property==1.5.1',
                      'tabulate==0.8.9',
                      'tornado==6.1.0',
                      'seaborn==0.11.2',
                      'scikit-learn==1.0.2',
                      'h5py==3.6.0',
                      'tables==3.7.0'
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.7',
    author='Carlos de Lannoy',
    author_email='carlos.delannoy@wur.nl',
    description='Supervise FRET event detection algorithms',
    long_description=readme(),
    license='MIT',
    keywords='FRET Forster resonance energy transfer supervised machine learning',
    url='https://github.com/cvdelannoy/FRETboard',
    entry_points={
        'console_scripts': [
            'FRETboard = FRETboard.__main__:main'
        ]
    },
    include_package_data=True
)
