import streamlit as st
from supabase import create_client

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)

response = supabase.table("papers").select("id").limit(1).execute()

print("Supabase connection successful!")
print(response.data)