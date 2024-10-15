import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key was loaded
if not google_api_key:
    st.error("API key not found. Please add it to your .env file as GOOGLE_API_KEY.")
else:
    # Streamlit app title
    st.title("Chatbot with Gemini Model")

    # Initialize the language model with the API key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, api_key=google_api_key)

    # System prompt for the chatbot
    system_prompt = (
        "You are an assistant for general question-answering tasks. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
    )
    system_message = SystemMessage(content=system_prompt)

    # User input
    query = st.chat_input("Say something: ")

    # Check if there is a query, then proceed with the chatbot response
    if query:
        human_message = HumanMessage(content=query)
        messages = [system_message, human_message]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.invoke(messages)

                # Extract only the content field from the response
                if hasattr(response, 'content'):  # Check if response has 'content' attribute
                    chatbot_reply = response.content
                else:
                    chatbot_reply = "Sorry, the response format was not as expected."

                # Display only the chatbot's reply content
                st.write(chatbot_reply)
                break  # Exit the loop if successful

            except Exception as e:
                st.write(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.write("Sorry, I couldn't complete your request due to a server error.")
