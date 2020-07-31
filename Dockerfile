FROM continuumio/miniconda3

MAINTAINER Carlos de Lannoy <carlos.delannoy@wur.nl>
RUN conda install -yc anaconda gcc_linux-64 && pip install git+https://github.com/cvdelannoy/FRETboard.git