# dexter-energy


## Prerequisites

- Docker
- Docker Compose

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd dexter-energy
```

### 2. Build and Run with Docker Compose

Use Docker Compose to build and run the containers defined in the ```docker-compose.yml``` file.

```bash
docker-compose up -d
```

To stop Docker containers use the following command.
```bash
docker-compose down
```

### 3. Accessing the Application

To access Web interfacess of services follow URLs:

- Airflow `http://localhost:8080`

Username: `admin`, Password: `admin`

Here you can trigger dags with configurations, see dag design, check dags'running status.

- MLflow `http://localhost:90/`

Here you can observe experiments and stored models.

### Configuration (Optional)

You can find `.env` files in folder `env_files/` to configure docker environment.

To adjust default model and airflow dags parameters you can edit files in `scripts/config/`.
