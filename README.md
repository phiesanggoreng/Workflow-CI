# Workflow-CI - Wine Quality ML Pipeline

## CI/CD Pipeline for Machine Learning Model Training & Deployment

### Structure
```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI workflow
├── MLProject/
│   ├── modelling.py                # Training script
│   ├── conda.yaml                  # Conda environment
│   ├── MLProject                   # MLflow project file
│   ├── wine_quality_preprocessing/ # Dataset
│   └── docker_hub_link.txt         # Docker Hub link
└── README.md
```

### CI Pipeline Steps
1. Checkout code
2. Setup Python 3.11
3. Install dependencies
4. Run MLflow Project training
5. Upload artifacts to GitHub
6. Build Docker image (`mlflow models build-docker`)
7. Push Docker image to Docker Hub

### Docker Hub
- Image: `brass17/wine-quality-model`
- URL: https://hub.docker.com/r/brass17/wine-quality-model

### Required Secrets
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

### Author
Nibras Ahmad Badruzzaman - Dicoding Submission
