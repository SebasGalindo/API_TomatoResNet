FROM python:3.13.2-slim

WORKDIR /app

COPY Resources /app/Resources

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "TomatoAPI:app"]
