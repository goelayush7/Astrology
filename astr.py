import os
from datetime import date, time
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env if available
load_dotenv()

st.set_page_config(page_title="AI Astrologer ✨ (LangChain)", page_icon="✨", layout="centered")

st.markdown(
    "<h1 style='text-align:center;'>AI Astrologer ✨</h1>"
    "<p style='text-align:center;color:gray;'>LLM-powered astrology profile + Q&A</p>",
    unsafe_allow_html=True
)

with st.expander("Disclaimer", expanded=False):
    st.write("This demo uses a Large Language Model to produce astrology-style guidance. "
             "It may be fictional or speculative and is not professional advice.")

# --------- Model Setup ----------
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.info("⚠️ Tip: Set your OPENAI_API_KEY in a .env file to get proper answers.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Profile generator chain
profile_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly professional astrologer. Build a short profile strictly from the user's birth details. "
     "If a detail is missing (like coordinates), infer broad insights without claiming precision. "
     "Avoid deterministic predictions. Keep it concise (120-180 words). "
     "Use sections: 'Key Themes', 'Strengths', 'Watch-outs', and end with one practical tip."),
    ("user",
     "Name: {name}\nDOB: {dob}\nTOB: {tob}\nPlace: {place}\n"
     "Create a concise natal-style overview.")
])
profile_chain = profile_prompt | llm | StrOutputParser()

# Q&A chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an astrologer answering one focused question. "
     "Use the provided profile for context and the birth details. "
     "Offer supportive, actionable guidance (100-150 words), avoid fatalistic claims, no medical/financial guarantees. "
     "End with 2-3 concrete next steps."),
    ("user",
     "Birth Details -> Name: {name}; DOB: {dob}; TOB: {tob}; Place: {place}\n"
     "Profile Context:\n{profile}\n"
     "Question: {question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

# --------- UI ----------
with st.form("birth_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name*", value="Ayush")
        dob = st.date_input("Date of Birth*", value=date(2004, 5, 25))
    with col2:
        tob = st.time_input("Time of Birth", value=time(9, 15))
        place = st.text_input("Place of Birth (City, Country)", value="New Delhi, India")
    submitted = st.form_submit_button("Generate AI Profile")

if "profile_text" not in st.session_state:
    st.session_state.profile_text = ""

if submitted:
    with st.spinner("🔮 Generating profile..."):
        st.session_state.profile_text = profile_chain.invoke({
            "name": name,
            "dob": dob.isoformat(),
            "tob": tob.strftime("%H:%M"),
            "place": place,
        })
    st.success("Profile generated!")

if st.session_state.profile_text:
    st.markdown("## ✨ Your AI Astrology Profile")
    st.write(st.session_state.profile_text)
    st.divider()
    st.subheader("Ask a Question")
    question = st.text_input("Example: How does my career look this year?")
    if st.button("Ask"):
        if question.strip():
            with st.spinner("🌙 Consulting the stars..."):
                answer = qa_chain.invoke({
                    "name": name,
                    "dob": dob.isoformat(),
                    "tob": tob.strftime("%H:%M"),
                    "place": place,
                    "profile": st.session_state.profile_text,
                    "question": question.strip()
                })
            st.markdown("### Response")
            st.write(answer)
        else:
            st.warning("Please type a question first.")

st.write("---")
st.caption("Built with Streamlit + LangChain + OpenAI")
