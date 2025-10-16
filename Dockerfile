# Use a lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app/

# Expose port (Flask default: 5000)
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=ui/app.py

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]