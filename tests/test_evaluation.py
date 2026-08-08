import mlflow

from src.models import evaluation


def test_configure_mlflow_defaults_to_local_tracking(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("USE_DAGSHUB", "0")

    class DummyDagshub:
        @staticmethod
        def init(*args, **kwargs):
            return None

    monkeypatch.setattr(evaluation, "dagshub", DummyDagshub)

    evaluation.configure_mlflow(tmp_path)

    assert mlflow.get_tracking_uri().startswith("file:")
