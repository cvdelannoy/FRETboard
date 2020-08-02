# Recommended way to set up FRETboard as server:
# - start up as interactive container: docker run -p 127.0.0.1:5102:XXXX/tcp -it fretboard /bin/bash
#   - Replace XXXX by the tcp port id you want to use
# - start up FRETboard: bokeh serve  /FRETboard/FRETboard --num-procs=1 --check-unused-sessions 1000 --port=5102 --address=0.0.0.0
# - detach: ctrl-p --> ctrl-q

FROM python:3.7

MAINTAINER Carlos de Lannoy <carlos.delannoy@wur.nl>

COPY . /FRETboard

RUN apt update && apt install -y build-essential
RUN pip install -e /FRETboard
