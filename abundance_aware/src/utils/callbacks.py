from transformers.trainer_callback import TrainerCallback
import mlflow

class LambdaATrackingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "get_lambda_a"):
            lambda_a = model.get_lambda_a()
            metrics = {}
            for layer_idx, layer_vals in enumerate(lambda_a):
                for head_idx, val in enumerate(layer_vals):
                    metrics[f"lambda_a/layer{layer_idx}_head{head_idx}"] = val.item()
            mlflow.log_metrics(metrics, step=state.global_step)

class ConditionerTrackingCallback(TrainerCallback):
    """Replaces LambdaATrackingCallback — logs every conditioner's
    gate diagnostics each eval, with no per-conditioner code to add."""
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not hasattr(model, "conditioner_logging_metrics"):
            return
        metrics = model.conditioner_logging_metrics()
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)