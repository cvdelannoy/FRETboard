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
                      'pandas==1.0.5',
                      'statsmodels==0.11.1',
                      'pomegranate==0.13.4',
                      'bokeh==2.4.2',
                      'cached-property==1.5.1',
                      'tabulate==0.8.3',
                      'tornado==6.0.3',
                      'seaborn==0.9.0',
                      'scikit-learn==0.21.2',
                      'h5py==2.10.0',
                      'tables==3.6.1',
                      'psutil==5.7.0'
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
