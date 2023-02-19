FROM python:3.10


WORKDIR /code

COPY . /code
COPY ./semantic_app /code/semantic_app
COPY ./artifacts /code/artifacts

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install
# RUN pip install --no-cache-dir -r requirements.txt

RUN ls -la

EXPOSE 8000 

CMD ["poetry", "run", "python", "semantic_app/app.py"]