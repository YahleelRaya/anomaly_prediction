from groq import Groq
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

print("API Key:", api_key)
client = Groq(api_key=api_key)

def get_investment_strategy(data):
    # Transform data to JSON or use directly if already prepared
    json_data = data.to_json(orient='records')

    # Update the system message to include explicit instruction for JSON format
    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst who provides investment strategies in JSON format. Analyze the provided data and suggest an investment strategy. Please return the response as a JSON object, including fields for 'investment_type', 'confidence_level', and 'strategy_description'."
        },
        {
            "role": "user",
            "content": json_data
        }
    ]

    # Attempt to call the Groq API and handle possible errors
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error fetching investment strategy: {e}")
        return None

# Example of how to use this function
# if __name__ == '__main__':
#     import pandas as pd
#     # Assume 'data' is a DataFrame loaded with the necessary market data
#     data = pd.read_csv('path_to_your_data.csv')
#     strategy = get_investment_strategy(data)
#     print(strategy)

