FROM unsloth/unsloth:latest

USER root

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY . .

RUN mkdir -p models results data/raw && chmod -R 777 models results data

ENTRYPOINT []
CMD ["bash"]
