FROM python:3.9
WORKDIR /code

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app

ENV MongoConString="mongodb+srv://st10066487:debtduty96@apds.sw61z.mongodb.net/?retryWrites=true&w=majority&appName=APDS"

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]