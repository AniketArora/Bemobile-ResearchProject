FROM python:3.8

RUN pip install --no-cache-dir -U pip

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /python-app
COPY ./python-app .

EXPOSE 8000
CMD ["python", "main.py"]