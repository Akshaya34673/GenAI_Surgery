# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy backend code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask runs on
EXPOSE 5000

# Command to run Flask app
CMD ["python", "app.py"]
