# PMLDL_A1

# Flat Price Prediction System

This project provides a flat price prediction service using a machine learning model based on a GradientBoostingRegressor. The service is deployed using two containers, one for a Flask API that serves predictions and another for a Gradio app for user interaction. The containers are managed using Docker Compose.

## Project Structure

```
├── code
│   ├── datasets              # Data preprocessing scripts
│   ├── deployment            # Deployment files for Flask API and Gradio App
│   │   ├── api               # Flask API for model inference
│   │   └── app               # Gradio app for user interface
│   ├── models                # Model training and saved models
├── data                      # Datasets and preprocessed files
├── models                    # Trained models saved as .pkl files
├── LICENSE                   # License file
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Installation and Setup

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-repo/flat-price-prediction.git
cd flat-price-prediction
```

### 2. Install Python Dependencies
Set up a virtual environment and install the required Python packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to retrain the model, you can run the training script:
```bash
python code/models/train_model.py
```

This will generate and save the trained model in the `models/` directory.

### 4. Set up Docker and Docker Compose
Ensure Docker and Docker Compose are installed on your system.

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)

### 5. Running the Project

#### Start the Containers
The project is composed of two services: 
- **flask-api**: Exposes a REST API to make predictions.
- **gradio-app**: A Gradio UI to interact with the model.

To run both services, use Docker Compose:

```bash
docker-compose -f code/deployment/docker-compose.yml up --build
```

This will build and start two containers:
- The Flask API on `http://172.20.0.2:5001/`
- The Gradio app on `http://0.0.0.0:5155/`

### 6. Access the Services

- **Gradio App**: Open a browser and go to `http://0.0.0.0:5155/`.
- **Flask API**: Use a tool like `curl` or Postman to make requests to the Flask API at `http://172.20.0.2:5001/`.

## API Endpoints

### Flask API

1. **`/` [GET]**
   - Returns a welcome message for the API.
  
2. **`/info` [GET]**
   - Provides metadata about the deployed model, including:
     - Best hyperparameters
     - R² score
  
3. **`/predict` [POST]**
   - Accepts JSON input to predict the flat price based on the provided features.

   Example response:
   ```json
   {
       "flat price": "250000 S$"
   }
   ```

## Gradio App

The Gradio app provides a user-friendly web interface to input flat details and view the predicted price. Access it via `http://0.0.0.0:5155/` once the containers are running.

## Model Training and Evaluation

The machine learning model used in this project is a GradientBoostingRegressor. The model was trained using the following features:
- **month**: Date of sale
- **town**: Location of the flat
- **flat_type**: Type of flat (e.g., 3-room, 4-room)
- **floor_area_sqm**: Size of the flat in square meters
- **remaining_lease**: Lease remaining on the flat

### Model Performance

After training, the model achieved the following metrics:
- **MAE (Mean Absolute Error)**: 12,000 S$
- **MSE (Mean Squared Error)**: 300,000,000 S$
- **R² Score**: 0.85
