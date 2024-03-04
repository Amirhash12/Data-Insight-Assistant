import streamlit as st
#from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
#from llama_index.llms import OpenAI
import openai
from llama_index.readers.web import SimpleWebPageReader
import pandas as pd
import numpy as np
import csv
from datetime import datetime

import plotly.express as px

from llama_index.core import SimpleDirectoryReader , VectorStoreIndex, ServiceContext

from llama_index.llms.openai import OpenAI

#from llama_index.llms import OpenAI
import openai
from llama_index.core import SummaryIndex
#from llama_index.readers.web import SimpleWebPageReader


def save_conversation(messages):
    # Define the filename, e.g., conversations.csv
    filename = 'conversations/conversations.csv'
    
    # Open the file in append mode
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Optionally, write headers if the file is empty/new
        # writer.writerow(["timestamp", "role", "content"])
        
        # Write each message to the CSV file
        for message in messages:
            # Include a timestamp for each entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, message['role'], message['content']])

def load_latest_conversation():
    filename = 'conversations.csv'
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            # Read the conversations and split them into messages
            reader = csv.reader(file)
            messages = list(reader)
            if messages:
                # Convert messages back to the required format
                return [{"timestamp": msg[0], "role": msg[1], "content": msg[2]} for msg in messages]
    except FileNotFoundError:
        # If the file doesn't exist, return an empty list
        return []
    return []


def display_conversations(messages, limit=5):
    # Display only the last 'limit' messages
    for message in messages[-limit:]:
        role = message["role"]
        time = message.get("timestamp", "Unknown Time")
        content = message["content"]
        
        # Customize the display for different roles
        if role == "user":
            st.container().markdown(f"**You ({time}):** {content}")
            
        elif role == "system":
            st.container().markdown(f"**Evaluation Result ({time}):** {content}")    
        else:
            st.container().markdown(f"**IIC AI Assistant ({time}):** {content}")
            


def main():
    
    st.sidebar.title("Menu")
    menu_options = ["Chat" , "Insights","Backend","GA4 Integration"]
    selected_option = st.sidebar.selectbox("Select an option", menu_options)
    
    if selected_option == "Chat":
        openai.api_key = st.secrets.openai_key
        st.header("Product Assistant !📚💬")

        # Initialize session state for showing conversations
        if "show_conversations" not in st.session_state:
            st.session_state.show_conversations = False

        if "messages" not in st.session_state.keys(): # Initialize the chat message history
            st.session_state.messages = [
                {"role": "assistant", "content": "Data Insight Assistant ! 🦾. Ask me anything. "},
            ]
            #load_latest_conversation()

        @st.cache_resource(show_spinner=False)
        def load_data():
            with st.spinner(text="Loading data from database – hang tight! This should take 1-2 minutes."):
                reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
                docs = reader.load_data()
                service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert assistant and your job is to answer technical questions. Keep your answers technical and based on facts – do not hallucinate features."))
                index = VectorStoreIndex.from_documents(docs, service_context=service_context)
                
                
                return index

        index = load_data()
        

   
   
   
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        

    


        if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})



        for message in st.session_state.messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])


        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
                    save_conversation(st.session_state.messages)

              
    elif selected_option == "Insights":
        st.header("Data Insights 📊")
        # Assuming your data files are located in a directory named 'data' within your Streamlit application directory
        data_directory = "data/"

        # Load each CSV file into a pandas DataFrame
        production_df = pd.read_csv(f"{data_directory}production.csv", parse_dates=['Date'])
        products_df = pd.read_csv(f"{data_directory}products.csv", parse_dates=['Launch_Date'])
        reviews_df = pd.read_csv(f"{data_directory}reviews.csv", parse_dates=['Date'])
        sales_df = pd.read_csv(f"{data_directory}sales.csv", parse_dates=['Date'])
        # Summary statistics for sales and production
        total_sales_quantity = sales_df['Quantity'].sum()
        total_production_quantity = production_df['Quantity'].sum()
        avg_rating = reviews_df['Rating'].mean()
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales Quantity", total_sales_quantity)
        col2.metric("Total Production Quantity", total_production_quantity)
        col3.metric("Average Product Rating", f"{avg_rating:.2f}")

        # Sales Insights
        st.subheader("Sales Insights")
        sales_by_product = sales_df.groupby('Product_ID')['Quantity'].sum().reset_index()
        sales_by_product = sales_by_product.merge(products_df[['Product_ID', 'Name']], on='Product_ID')
        fig_sales = px.bar(sales_by_product, x='Name', y='Quantity', title="Sales by Product")
        st.plotly_chart(fig_sales)

        # Production Insights
        st.subheader("Production Insights")
        production_by_product = production_df.groupby('Product_ID')['Quantity'].sum().reset_index()
        production_by_product = production_by_product.merge(products_df[['Product_ID', 'Name']], on='Product_ID')
        fig_production = px.bar(production_by_product, x='Name', y='Quantity', title="Production by Product")
        st.plotly_chart(fig_production)

        # Review Sentiments
        st.subheader("Review Sentiments")
        sentiment_counts = reviews_df['Sentiment'].value_counts().reset_index()
        fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='count', title="Review Sentiments Distribution")
        st.plotly_chart(fig_sentiment)

        # Interactive Product Selector for Detailed Insights
        st.subheader("Detailed Product Insights")
        product_list = products_df['Name'].sort_values().unique()
        selected_product = st.selectbox("Select a Product:", product_list)

        if selected_product:
            product_id = products_df[products_df['Name'] == selected_product]['Product_ID'].iloc[0]
            product_sales = sales_df[sales_df['Product_ID'] == product_id]
            product_reviews = reviews_df[reviews_df['Product_ID'] == product_id]

            # Display selected product sales over time
            fig_product_sales = px.line(product_sales, x='Date', y='Quantity', title=f"Sales Over Time: {selected_product}")
            st.plotly_chart(fig_product_sales)
            # Assuming product_sentiment_counts is structured similarly to sentiment_counts
            product_sentiment_counts = product_reviews['Sentiment'].value_counts().reset_index()
            product_sentiment_counts.columns = ['Sentiment', 'count']  # Rename columns to match expected names

            st.plotly_chart(fig_product_sentiment)
        
    elif selected_option == "Backend":
        st.header("Behind the Scenes 🛠️")
        st.subheader("How my Code Works?")
        st.markdown("""
        Dive into the backend operations that power this application. Understand the data processing, AI interactions, and visualizations that deliver a seamless user experience.
        """)
        
        st.subheader("Technical Workflow")
        st.markdown("""
        **1. Data Processing:** Data from various CSV files is loaded and processed using Pandas, allowing for complex data manipulation and analysis.
        
        **2. AI-Powered Chat:** Leveraging OpenAI's GPT model and `llama_index` libraries, the application offers an intelligent chat interface. Messages are saved and managed efficiently, providing a dynamic conversation history.
        
        **3. Insights Visualization:** Utilizing Plotly, the application creates interactive charts and graphs for data insights, offering a rich, interactive user experience.
        
        **4. Backend Integration:** The app integrates various backend operations seamlessly, from data loading to AI interactions, ensuring a robust and responsive platform.
        """)

        st.subheader("Technologies Used")
        col1, col2, col3 = st.columns(3)
        col1.metric("Data Analysis", "Pandas")
        col2.metric("AI Model", "OpenAI GPT")
        col3.metric("Visualization", "Plotly")
        
        st.subheader("Interactive Demonstration")
        st.markdown("""
        Select any technology to learn more about how it contributes to the application.
        """)
        tech_options = ["Pandas", "OpenAI GPT", "Plotly"]
        selected_tech = st.selectbox("Choose a technology:", tech_options)
        
        if selected_tech == "Pandas":
            st.info("Pandas is a powerful data analysis and manipulation library for Python, providing fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive.")
        elif selected_tech == "OpenAI GPT":
            st.info("OpenAI's GPT models are state-of-the-art natural language processing models that can generate human-like text, answer questions, summarize content, and more, enabling advanced AI-powered interactions in applications.")
        elif selected_tech == "Plotly":
            st.info("Plotly is a graphing library that makes interactive, publication-quality graphs online. It offers a wide range of chart types and styles, and integrates with many programming languages and frameworks for data science and analytics.")

    elif selected_option == "GA4 Integration":
        st.header("GA4 Integration Insights 📈")
        st.markdown("""
        Explore how integrating Google Analytics 4 (GA4) can enhance the understanding of user interactions within this app. Below are simulated insights based on typical GA4 analytics.
        """)
        
        st.subheader("User Engagement")
        # Simulate user engagement data
        engagement_data = {
            "Page Views": 1200,
            "Unique Visits": 800,
            "Avg. Session Duration (min)": 5.2,
            "Bounce Rate (%)": 40
        }
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Page Views", engagement_data["Page Views"])
        col2.metric("Unique Visits", engagement_data["Unique Visits"])
        col3.metric("Avg. Session Duration", engagement_data["Avg. Session Duration (min)"])
        col4.metric("Bounce Rate", engagement_data["Bounce Rate (%)"])
        
        st.subheader("Traffic Sources")
        # Simulate traffic source data
        traffic_sources = pd.DataFrame({
            "Source": ["Direct", "Referral", "Social", "Organic Search"],
            "Users": [300, 250, 150, 100]
        })
        fig_traffic = px.pie(traffic_sources, names='Source', values='Users', title="User Traffic Sources")
        st.plotly_chart(fig_traffic)
        
        st.subheader("User Actions")
        st.markdown("""
        Display user actions within the app, such as button clicks, form submissions, and other interactions, to demonstrate how GA4 can track and analyze user behavior.
        """)
        # Example user action data
        actions_data = {
            "Chat Messages Sent": 500,
            "Insight Reports Generated": 300,
            "Product Searches": 200
        }
        fig_actions = px.bar(x=list(actions_data.keys()), y=list(actions_data.values()), title="User Actions")
        st.plotly_chart(fig_actions)
        
        st.subheader("Implementing GA4")
        st.markdown("""
        To integrate GA4 with your Streamlit app, follow these steps:
        1. Create a GA4 property in your Google Analytics account.
        2. Obtain the "Measurement ID" from your GA4 property settings.
        3. Use a suitable library or API to send events from your Streamlit app to GA4.
        4. Analyze the collected data in your GA4 dashboard to gain insights into user behavior and app performance.
        """)




if __name__ == "__main__":
    main()
