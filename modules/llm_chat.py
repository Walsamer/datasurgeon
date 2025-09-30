"""
LLM integration for chat-based functions.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env file

class ChatModule:
    """
    chat based LLM interaction module using google gemini
    """

    def __init__(self, dataframe_context: Optional[dict] = None, plot_context: Optional[dict] = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Please add your GOOGLE_API_KEY to a .env file in the project root.")
        
        #configure gemini:
        genai.configure(api_key=self.api_key)
        self.model=genai.GenerativeModel("gemini-2.5-flash")

        #store context
        self.current_df = None
        self.dataframe_context = dataframe_context
        self.plot_context = plot_context
        
        print("ChatModule initialized with Gemini model.")

    def set_dataframe(self, df: pd.DataFrame):
        """Set the current dataframe and update context."""
        self.current_df = df
        print("Set Dataframe.")
    
    def analyze_dataframe(self, df=None):
        """
        Dataframe - Metadata for LLM-Kontext
        """
        if df is None:
            df = self.current_df
    
        if df is None:
            return {}
        
        context = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_statistics": df.describe().to_dict()
        }
        return context
    
    def chat(self, message: str) -> str:
        """
        Send a message to the LLM and get a response.
        """
        try:
            if self.current_df is not None:
                context = self.analyze_dataframe()
                full_prompt = f"""
                You are an expert assistant in explorative data analysis chatting with a User. Based on your experience in dealing with data and the given dataset, help the user to reach his goal. Here is the context of the current dataframe:
                {context}

                User message: {message}

                Please provide a helpful and concise response based on the dataframe context.
                """
            else:
                full_prompt = f"""
                You are a data analysis assistant chatting with a User.
                Note: No dataframe is currently set, therefore make some funny jokes about data and missing data.
                
                User message: {message}

                Please provide a helpful and concise response.
                """
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error during chat: {e}"


    # def interpret_plot(self, fig):
    #     """
    #     Analyze Plotly configuration for LLM Kontext
    #     """
    #     return {
    #         "plot_type": plot_config.get("type"),
    #         "data_summary": plot_config.get("data_summary")
    #     }
    

#test functoin
def test_chat_module():
    try:
        chat = ChatModule()
        response = chat.chat("Hello! Can you help me analyze data?")
        print("Response:", response)
        return True
    except Exception as e:
        print("Error during test:", e)
        return False
    
if __name__ == "__main__":
    print("Testing ChatModule...")
    test_chat_module()