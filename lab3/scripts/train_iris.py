import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os


os.environ['REQUESTS_CA_BUNDLE'] = '/home/user/lab3/ssl/certs/ca.crt'
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
mlflow.set_experiment("Iris Classification")

def train_iris_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print("Начало обучения модели")
    print(f"Датасет: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Классы: {list(target_names)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    #параметры для экспериментов
    experiments = [
        {"n_estimators": 50, "max_depth": 3, "run_name": "RandomForest_50_3"},
        {"n_estimators": 100, "max_depth": 5, "run_name": "RandomForest_100_5"}, 
        {"n_estimators": 200, "max_depth": None, "run_name": "RandomForest_200_None"}
    ]
    
    results = []
    
    for params in experiments:
        print(f"\nОбучение с параметрами: {params}")
        
        with mlflow.start_run(run_name=params["run_name"]):
            #логирование параметров
            mlflow.log_params(params)
            mlflow.log_param("test_size", 0.3)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("dataset", "Iris")

            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            #логирование метрик
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            #логирование модели
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_model",
                registered_model_name="IrisClassifier"
            )
            
            #логирование информации о данных
            mlflow.log_text(str(iris.DESCR), "dataset_info.txt")
            mlflow.log_dict({
                "feature_names": feature_names,
                "target_names": target_names.tolist(),
                "data_shape": X.shape
            }, "dataset_metadata.json")
            
            results.append({
                "run_name": params["run_name"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            print(f"Завершено: {params['run_name']}")
            print(f"Точность: {accuracy:.4f}")
            print(f"F1-score: {f1:.4f}")
    
    #сравнение результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:")
    print("="*60)
    best_run = max(results, key=lambda x: x['accuracy'])
    
    for result in results:
        marker = " успешно" if result == best_run else ""
        print(f"{result['run_name']}:")
        print(f"  Accuracy: {result['accuracy']:.4f}{marker}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f}")
        print()

    print(f"Лучшая модель: {best_run['run_name']} (Accuracy: {best_run['accuracy']:.4f})")
    return results, best_run


if __name__ == "__main__":
    try:
        results, best_run = train_iris_model()
        print("\nВсе эксперименты успешно завершены и залогированы в MLflow!")
    except Exception as e:
        print(f"Ошибка: {e}")
