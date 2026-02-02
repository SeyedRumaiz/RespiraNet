from abc import ABC, abstractmethod
import pandas as pd
import streamlit as st

class Visualizer(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        pass


class ConfidenceBarChart(Visualizer):
    def plot(self, confidence: float):
        """
        Plots confidence for NORMAL vs PNEUMONIA
        """
        
        probs = {"NORMAL": 1-confidence, "PNEUMONIA": confidence}
        df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
        st.bar_chart(df.set_index("Class"))
