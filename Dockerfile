FROM ubuntu:22.04

WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y python3-pip curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock .
RUN uv sync --frozen --no-dev

COPY model.skops .
COPY preprocessor.pkl .
COPY app/ app/
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]