from mlflow.tracking import MlflowClient

client = MlflowClient("http://127.0.0.1:5000")

run = client.get_run("de9bda5ca49a48c29c44ebeb48136832")

print("Run name:", run.data.tags.get("mlflow.runName"))
print("Status:", run.info.status)
print("Metrics:", run.data.metrics)