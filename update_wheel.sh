#!/bin/bash
# Convenience script to update wheels
rm -rf dist build/ FRET_board.egg-info/
python setup.py sdist
python setup.py bdist_wheel
