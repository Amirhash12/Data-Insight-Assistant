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

from llama_index.core import SimpleDirectoryReader , VectorStoreIndex, ServiceContext , Settings

from llama_index.llms.openai import OpenAI

#from llama_index.llms import OpenAI
import openai
from llama_index.core import SummaryIndex
#from llama_index.readers.web import SimpleWebPageReader

# Update the Settings as per the documentation provided
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)



def save_conversation(messages):
    filename = 'conversations/conversations.csv'
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for message in messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, message['role'], message['content']])



def load_latest_conversation():
    filename = 'conversations.csv'
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            messages = list(reader)
            return [{"timestamp": msg[0], "role": msg[1], "content": msg[2]} for msg in messages]
    except FileNotFoundError:
        return []


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')





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
            
import io
import pandas as pd

def safe_read_csv(file_path, encoding='ISO-8859-1', errors='replace'):
    with open(file_path, 'r', encoding=encoding, errors=errors) as file:
        content = file.read()
    
    # Handling special characters based on the provided dictionaries
    special_chars_mappings = {
        '®': '(R)',    # Registered trademark symbol
        '\x99': '',    # TM symbol in Windows-1252
        'Ö': 'O',      # Umlaut O
        '\xa0': ' ',   # Non-breaking space
        '\x93': '"',   # Left double quotation mark in Windows-1252
        '\x94': '"',   # Right double quotation mark in Windows-1252
        '°': ' degrees ', # Degree symbol
        '\x96': '-',   # Dash in Windows-1252
        '\x80': 'EUR', # Euro symbol in Windows-1252
    }
    
    for char, replacement in special_chars_mappings.items():
        content = content.replace(char, replacement)

    # Convert the cleaned string back into a pandas DataFrame
    return pd.read_csv(io.StringIO(content))


    
def main():
    
    st.sidebar.title("Menu")
    menu_options = ["Chat" , "Insights","Backend","Price Adjustment","Geographical Pricing Analysis"]
    selected_option = st.sidebar.selectbox("Select an option", menu_options)
    
    # Assuming your data files are located in a directory named 'data' within your Streamlit application directory
    data_directory = "data/"
    # Load each CSV file into a pandas DataFrame here, so it's available for all sections
    fx_rates_df = safe_read_csv(f"{data_directory}fx_rates.csv")
    sales_price_df = safe_read_csv(f"{data_directory}sales_price.csv")
    item_names_df = safe_read_csv(f"{data_directory}item_names.csv")
    Merge_file_df = safe_read_csv(f"{data_directory}Merge_file.csv")
    regulizations_df = safe_read_csv(f"{data_directory}regulizations.csv")
   
    # Proceed to perform necessary data manipulations that are common across different sections

    if selected_option == "Chat":
        openai.api_key = st.secrets.openai_key
        st.header("Product and Pricing Assistant !📚💬")

        # Initialize session state for showing conversations
        if "show_conversations" not in st.session_state:
            st.session_state.show_conversations = False

        if "messages" not in st.session_state.keys(): # Initialize the chat message history
            st.session_state.messages = [
                {"role": "assistant", "content": "Assistant ! 🦾. Ask me anything. "},
            ]
        
        @st.cache_resource(show_spinner=False)
        def load_data():
            with st.spinner(text="Loading data from database – hang tight! This should take 1-2 minutes."):
                reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
                docs = reader.load_data()
                index = VectorStoreIndex.from_documents(docs)
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
        fx_rates_df = pd.read_csv(f"{data_directory}fx_rates.csv", encoding='ISO-8859-1')
        sales_price_df = pd.read_csv(f"{data_directory}sales_price.csv", encoding='ISO-8859-1')
        item_names_df = pd.read_csv(f"{data_directory}item_names.csv", encoding='ISO-8859-1')
        
        # Merge the sales_price_df with the item_names_df to get the product names
        sales_price_df = sales_price_df.merge(item_names_df, on='Item_No_', how='left')
        # Assume fx_rates_df contains columns 'Currency' and 'Rate_to_USD'
        # Convert prices to a single currency (USD) for comparison/visualization
        sales_price_df = sales_price_df.merge(fx_rates_df, left_on='Currency', right_on='Currency', how='left')
        sales_price_df['Price'] = pd.to_numeric(sales_price_df['Price'], errors='coerce')
        sales_price_df['Fixed_rate'] = pd.to_numeric(sales_price_df['Fixed_rate'], errors='coerce')  # Adjust column name as needed
        sales_price_df['Price_in_USD'] = sales_price_df['Price'] * sales_price_df['Fixed_rate']

        # Visualization of Product Pricing Insights
        st.subheader("Product Pricing Insights")
        fig_pricing = px.bar(sales_price_df, x='PN_name', y='Price_in_USD', color='Currency', title="Product Pricing in USD")
        st.plotly_chart(fig_pricing)

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

    
    elif selected_option == "Price Adjustment":
        st.header("Multiple Product Price Adjustment 🔄")
        # Initialize session state to store selected products and their price adjustments
        if 'adjustments' not in st.session_state:
            st.session_state.adjustments = []
        if 'adjustment_index' not in st.session_state:
            st.session_state.adjustment_index = 0  # Initialize the index


        # Container to hold dynamic input forms
        adjustment_container = st.container()

        # Function to add a new product adjustment entry
        def add_product_adjustment():
          st.session_state.adjustments.append({'product': None, 'percentage': 0})
          st.session_state.adjustment_index += 1  # Increment the index for unique keys

        # Button to add a new product-price adjustment form
        st.button("Add Product for Adjustment", on_click=add_product_adjustment)
        
        # Display forms for each product adjustment
        for idx, adjustment in enumerate(st.session_state.adjustments):
            with adjustment_container:
                cols = st.columns(2)
                # Use the index in the session state to generate unique keys
                product_key = f"product_{idx}_{st.session_state.adjustment_index}"
                percentage_key = f"percentage_{idx}_{st.session_state.adjustment_index}"
              
                product_selection = cols[0].selectbox("Select a Product:", [''] + list(Merge_file_df['Item names'].unique()), key=product_key)
                percentage_input = cols[1].number_input("Adjustment Percentage:", value=0, format="%d", key=percentage_key)
                # Update session state
                adjustment['product'] = product_selection
                adjustment['percentage'] = percentage_input


     # Apply adjustments and visualize
        if st.button("Apply Price Adjustments"):
        # Create a copy to modify prices
          adjusted_df = Merge_file_df.copy()

           # Apply adjustments
          for adjustment in st.session_state.adjustments:
             if adjustment['product']:
                 adjusted_df.loc[adjusted_df['Item names'] == adjustment['product'], 'New_Price'] = adjusted_df['Price'] * (1 + (adjustment['percentage'] / 100.0))

          # Visualization of adjusted prices
          if not adjusted_df.empty:
              fig = px.bar(adjusted_df, x='Item names', y=['Price', 'New_Price'], color='Currency', barmode='group', title="Price Comparison After Adjustments")
              st.plotly_chart(fig)
             # Step 2: Add Download Button
              csv = convert_df_to_csv(adjusted_df)  # Convert adjusted DataFrame to CSV
              st.download_button(
                label="Download adjusted price data as CSV",
                data=csv,
                file_name='adjusted_price_data.csv',
                mime='text/csv',
                )
          else:
              st.write("No adjustments to apply.")


    elif selected_option == "Geographical Pricing Analysis":
        st.header("Geographical Pricing Analysis 🌍")

        # Allow users to select one or multiple products
        selected_products = st.multiselect('Select Products', options=Merge_file_df['Item names'].unique())

        # Filter the DataFrame based on selected products
        if selected_products:
            filtered_df = Merge_file_df[Merge_file_df['Item names'].isin(selected_products)]
        else:
         filtered_df = pd.DataFrame()  # An empty DataFrame if no product is selected

        # Visualization
        if not filtered_df.empty:
         fig = px.bar(filtered_df, x='Country_Code', y=' New Price in EUR ', color='Item names', barmode='group',
                     title='Geographical Pricing Analysis', labels={' New Price in EUR ': 'New Price in EUR'})
         st.plotly_chart(fig)
        else:
         st.write("Please select at least one product to see the pricing analysis.")





if __name__ == "__main__":
    main()
