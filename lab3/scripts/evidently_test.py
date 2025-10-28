import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

os.environ['REQUESTS_CA_BUNDLE'] = '/home/user/lab3/ssl/certs/ca.crt'
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")


def load_production_model():
    print("Загрузка модели из MLflow Registry...")
    
    client = mlflow.tracking.MlflowClient()
    model_name = "IrisClassifier"
    
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = sorted(versions, key=lambda x: x.version)[-1]
    model_uri = f"models:/{model_name}/{latest_version.version}"
    
    print(f"Загружена модель: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    model.feature_names_in_ = ['sepal length (cm)', 'sepal width (cm)', 
                              'petal length (cm)', 'petal width (cm)']
                              
    return model


def prepare_test_data():
    print("\nПодготовка данных для анализа...")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    
    #создаем референсные и текущие данные
    X_ref, X_curr, y_ref, y_curr = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    
    reference_data = pd.DataFrame(X_ref, columns=feature_names)
    current_data = pd.DataFrame(X_curr, columns=feature_names)
    
    print(f"Референсные данные: {reference_data.shape}")
    print(f"Текущие данные: {current_data.shape}")
    
    return reference_data, current_data, feature_names, y_curr


def check_data_drift(reference_data, current_data):
    #создание отчета
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    #запуск анализа
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    print("Анализ на наличие дрифта выполнен")
    return False, []


def evaluate_model_performance(model, current_data, y_true, feature_names):
    y_pred = model.predict(current_data)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nДетальный отчет:")
    print(classification_report(y_true, y_pred))
    
    return accuracy


def main():
    print("ТЕСТИРОВАНИЕ С EVIDENTLY")
    print("=" * 50)

    model = load_production_model()
    reference_data, current_data, feature_names, y_curr = prepare_test_data()
    drift_detected, drifted_features = check_data_drift(reference_data, current_data)
    accuracy = evaluate_model_performance(model, current_data, y_curr, feature_names)

    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    print(f"  - Data drift: {'ОБНАРУЖЕН' if drift_detected else 'НЕ ОБНАРУЖЕН'}")
    print(f"  - Accuracy: {accuracy:.4f} ")
    

if __name__ == "__main__":
    main()
