from inference_engine import StreamingInferenceEngine

engine = StreamingInferenceEngine()

# Simulate first 100 predictions
engine.run(steps=100)

df_preds = engine.get_prediction_dataframe()
print(df_preds.head())
