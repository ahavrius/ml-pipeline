FROM apache/airflow:2.6.3-python3.10


ENV PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2

WORKDIR /opt/airflow

COPY requirements.txt .

RUN pip install -r requirements.txt

# RUN pip install "poetry==$POETRY_VERSION"

# COPY poetry.lock pyproject.toml ./

# RUN poetry install --only main --no-interaction  --no-root

# COPY . .

# EXPOSE 8080 5555 8793

# RUN airflow db init
# RUN airflow db init && airflow users create --firstname admin --lastname admin --email admin --password admin --username admin --role Admin

# CMD ["airflow", "webserver", "--port", "8080"]

# CMD ["sh", "./entrypoint.sh"]
# ENTRYPOINT ["/usr/bin/dumb-init", "sh", "entrypoint.sh"]
