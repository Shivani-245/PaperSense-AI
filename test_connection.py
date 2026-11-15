
import streamlit as st
from supabase import create_client

st.title("PaperSense AI - Supabase Connection Test")

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

    supabase = create_client(url, key)

    # Test database access
    response = (
        supabase.table("papers")
        .select("id")
        .limit(1)
        .execute()
    )

    st.success("Supabase connection successful!")
    st.write("Database connection is working.")

except Exception as e:
    st.error("Connection failed.")
    st.code(str(e))
