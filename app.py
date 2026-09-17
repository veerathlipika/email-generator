import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Email Generator",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Get API Key
# -----------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()


# -----------------------------
# LangChain Gemini Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


# -----------------------------
# LangChain Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional email writing assistant.

Write clear, polite, professional and grammatically correct emails.

Do not add unnecessary information.
Return only the email content."""
    ),
    (
        "human",
        """Write a professional email using the following details:

Recipient: {recipient}
Purpose: {purpose}
Tone: {tone}
Additional details: {details}

Include:
- Subject
- Greeting
- Main email body
- Professional closing"""
    )
])


# -----------------------------
# Create LangChain Chain
# -----------------------------
email_chain = prompt | llm


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📧 AI Email Generator")

st.write(
    "Generate professional emails using LangChain and Gemini."
)


# -----------------------------
# User Inputs
# -----------------------------
recipient = st.text_input(
    "👤 Recipient",
    placeholder="Example: Professor / HR Manager / Team Lead"
)


purpose = st.text_area(
    "📝 Email Purpose",
    placeholder="Example: Request leave for two days"
)


tone = st.selectbox(
    "🎯 Email Tone",
    [
        "Professional",
        "Formal",
        "Friendly",
        "Polite",
        "Casual"
    ]
)


details = st.text_area(
    "📌 Additional Details",
    placeholder="Enter any additional information you want to include..."
)


# -----------------------------
# Generate Email
# -----------------------------
if st.button("✨ Generate Email"):

    # Check required inputs
    if not recipient or not purpose:
        st.warning(
            "Please enter the recipient and email purpose."
        )
        st.stop()

    # Generate email
    with st.spinner("Generating email..."):

        try:
            response = email_chain.invoke({
                "recipient": recipient,
                "purpose": purpose,
                "tone": tone,
                "details": details
            })

            # ---------------------------------
            # Extract only the actual text
            # ---------------------------------
            email_text = response.content

            if isinstance(email_text, list):

                text_parts = []

                for item in email_text:

                    if isinstance(item, dict):

                        if item.get("type") == "text":
                            text_parts.append(
                                item.get("text", "")
                            )

                    elif isinstance(item, str):
                        text_parts.append(item)

                email_text = "\n".join(text_parts)

            # Make sure the result is a string
            email_text = str(email_text).strip()

            # ---------------------------------
            # Display Generated Email
            # ---------------------------------
            st.subheader("📨 Generated Email")

            st.text_area(
                "Email",
                value=email_text,
                height=350
            )

        except Exception as e:

            st.error(
                f"Error generating email: {e}"
            )
