FROM python:3.12.12-slim
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/openrsi-author
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN python tools.py verify && chmod -R go-rwx cases/*/judge
ENTRYPOINT ["python", "tools.py"]
