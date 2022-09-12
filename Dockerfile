# Recommended way to set up FRETboard as server:
# - start up as interactive container: docker run -p 127.0.0.1:5102:5102/tcp -it fretboard /bin/bash
# - start up FRETboard: bokeh serve  /FRETboard/FRETboard --num-procs=1 --check-unused-sessions 1000 --port=5102 --address=0.0.0.0
# - detach: ctrl-p --> ctrl-q

FROM python:3.7

MAINTAINER Carlos de Lannoy <carlos.delannoy@wur.nl>

COPY . /FRETboard

RUN apt update && apt install -y build-essential screen
RUN pip install -e /FRETboard

EXPOSE 5102
EXPOSE 80

CMD bokeh serve \
    --allow-websocket-origin="*" \
    --index=/FRETboard/FRETboard/templates/index.html \
    --num-procs=1 \
    --check-unused-sessions=1000 \
    --port=5102 \
    /FRETboard/FRETboard
