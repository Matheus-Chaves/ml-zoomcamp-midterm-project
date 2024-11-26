FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install --no-dev

EXPOSE 9696

CMD ["poetry", "run", "python", "predict.py"]
