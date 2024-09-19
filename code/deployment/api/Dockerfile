# Use the official Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Expose the port that the Flask app runs on
EXPOSE 5001

# Define environment variable for Flask
ENV FLASK_APP=code/deployment/app/app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
