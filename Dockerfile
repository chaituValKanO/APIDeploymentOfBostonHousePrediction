FROM python:3.7.6-stretch

# Copy our application code
WORKDIR /var/app


COPY . .
COPY requirements.txt .
# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Expose port
EXPOSE 5000

WORKDIR /var/app/app

# Start the app
CMD ["gunicorn", "predict:app", "--bind", "0.0.0.0:5000"]