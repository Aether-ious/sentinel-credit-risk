def test_model_output_shape():
    # Ensure model returns a probability between 0 and 1
    model = load_model()
    pred = model.predict_proba(dummy_data)[0][1]
    assert 0 <= pred <= 1