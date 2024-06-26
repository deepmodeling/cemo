FROM stevenyhw/dpemc:0.0.7.3-pytorch1.12.1-cuda11.6
COPY ./cemo /root/repo/cemo
WORKDIR /root/repo/cemo
RUN pip install -e .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
