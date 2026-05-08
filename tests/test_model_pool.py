def test_model_pool_factories_return_distinct_instances():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    pytest.importorskip("torchtune")

    from voxtream.utils.model import MODEL_POOL

    first = MODEL_POOL["phone_former"]()
    second = MODEL_POOL["phone_former"]()

    assert first is not second
