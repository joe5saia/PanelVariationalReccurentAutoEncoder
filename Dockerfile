FROM julia:1.5

COPY *.toml /app/ 
WORKDIR /app/

ENV JULIA_PROJECT=/app/
ENV JULIA_NUM_THREADS=10

RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

CMD julia