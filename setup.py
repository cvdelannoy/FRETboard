from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='FRET-board',
    version='0.0.4',
    packages=['FRETboard'],
    install_requires=['numpy==1.17.3',
                      'pandas==0.25.3',
                      'pomegranate==0.11.2',
                      'bokeh==1.4.0',
                      'cached-property==1.5.1',
                      'tabulate==0.8.3',
                      'tornado==6.0.3',
                      'seaborn==0.9.0',
                      'scikit-learn==0.21.2'],
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
