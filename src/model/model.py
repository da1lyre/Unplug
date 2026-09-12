# src/model/unplug_predictor.py
"""
Unplug - Smartphone Addiction Predictor
Использует Decision Tree модель напрямую
"""

import joblib
from pathlib import Path


class UnplugPredictor:
    """
    Предиктор зависимости на основе Decision Tree
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent.parent / "models" / "tree" / "decision_tree_v1.pkl"

        self.model = joblib.load(model_path)
        self.features = ['social_media_hours', 'daily_screen_time_hours', 'weekend_screen_time']

    def predict(self, daily_screen_time, weekend_screen_time, social_media_hours):
        """
        Предсказание зависимости

        Args:
            daily_screen_time: экранное время в будни (часы/день)
            weekend_screen_time: экранное время в выходные (часы/день)
            social_media_hours: время в соцсетях (часы/день)

        Returns:
            dict: {'addicted': bool, 'message': str, 'probability': float}
        """
        import pandas as pd

        # Формируем DataFrame
        user_data = pd.DataFrame([[
            social_media_hours,
            daily_screen_time,
            weekend_screen_time
        ]], columns=self.features)

        # Предсказание
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

    def get_importance(self):
        """Важность признаков"""
        return dict(zip(self.features, self.model.feature_importances_))