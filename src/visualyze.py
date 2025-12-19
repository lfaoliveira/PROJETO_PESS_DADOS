import subprocess

def see_model():
    subprocess.Popen(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--default-artifact-root",
            "./mlruns",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
    )


if __name__ == "__main__":
    see_model()
