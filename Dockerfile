FROM ubuntu:22.04

RUN apt-get -y update && \
    apt-get install -y python3-pip curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock .
RUN uv sync --frozen --no-dev

COPY app ./app
RUN chmod +x app/run.sh
COPY train.py .
COPY src ./src

EXPOSE 8000

CMD ["bash", "-c", "./app/run.sh"]