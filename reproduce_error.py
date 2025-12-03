try:
    from models.time_series import gnosis_lstm_forecaster

    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
