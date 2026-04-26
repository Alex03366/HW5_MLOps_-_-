# MLOps Pipeline: Iris Classification

## Цель проекта
Создание минимального воспроизводимого MLOps-контура для классификации цветов Iris с использованием DVC, MLflow и Feast.
## Структура проекта
├── data/
│ ├── raw/ # Сырые данные (DVC)
│ └── processed/ # Обработанные данные
├── models/ # Модели
├── prepare.py # Подготовка данных
├── train.py # Обучение модели
├── dvc.yaml # DVC пайплайн
├── params.yaml # Параметры
├── requirements.txt # Зависимости
└── README.md
## Как запустить
```bash
git clone https://github.com/Alex03366/HW5_MLOps_-_-
cd HW5_MLOps_-_-
pip install -r requirements.txt
dvc pull
dvc repro
dvc metrics show

## Где смотреть UI MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
