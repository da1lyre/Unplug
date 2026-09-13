"""
Unplug - Smartphone Addiction Predictor
"""
from pathlib import Path
import joblib
import pandas as pd

class UnplugPredictor:
    """
    The main predictor class.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent.parent / "models" / "tree" / "decision_tree_v1.pkl"

        self.model = joblib.load(model_path)
        self.features = ['social_media_hours', 'daily_screen_time_hours', 'weekend_screen_time']

    def predict(self, daily_screen_time, weekend_screen_time, social_media_hours):
        """
        Predicting addiction.

        Args:
            daily_screen_time: float
            weekend_screen_time: float
            social_media_hours: float

        Returns:
            dict: {'addicted': bool, 'message': str, 'probability': float}
        """

        user_data = pd.DataFrame([[
            social_media_hours,
            daily_screen_time,
            weekend_screen_time
        ]], columns=self.features)

        prediction = self.model.predict(user_data)[0]
        probability = self.model.predict_proba(user_data)[0]

        if prediction == 1:
            return {
                'addicted': True,
                'message': "⚠️ У вас высокая зависимость от смартфона!",
                'result': "ЗАВИСИМ",
                'probability': float(probability[1])
            }
        else:
            return {
                'addicted': False,
                'message': "✅ У вас здоровые привычки использования смартфона!",
                'result': "НЕ ЗАВИСИМ",
                'probability': float(probability[0])
            }
