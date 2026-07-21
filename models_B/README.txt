Trained models land here.

You must copy the GAN generator into this folder yourself:
    cp /path/to/generator_net_MPEA.pt models_B/

Everything else (.joblib files, metrics.json, feature_config.json) is
created by the matching training script:
    models_A -> step2_retrain_models_A.py
    models_B -> step3_retrain_models_B.py
    models_C -> step4_retrain_models_C.py
