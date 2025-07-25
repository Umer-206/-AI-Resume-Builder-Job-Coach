import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pypdf import PdfReader

# Set up page
st.set_page_config(page_title="Resume Coach", layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: 600;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
    }
    .stTextInput>div>div>input {
        font-weight: 500;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
st.sidebar.title("Resume Coach")
menu = st.sidebar.selectbox("📌 Select Feature", [
    "Resume Generator", 
    "Cover Letter Generator",
    "CV Analyzer", 
    "LinkedIn Summary Generator",
    "Mock Interview Chatbot"
])

# Load model
st.info("🔄 Model loading...")
llm = Ollama(model="gemma3:latest")
st.success("✅ Model loaded successfully.")

st.markdown("## 🧾 Resume Builder + Job Coach")

# Resume Generator
if menu == "Resume Generator":
    st.subheader("📄 Resume Generator")
    job_title = st.text_input("🔹 Job Title")
    experience = st.text_area("🔹 Work Experience Summary")
    skills = st.text_area("🔹 List of Skills (comma separated)")

    if st.button("✍️ Generate Resume"):
        prompt = PromptTemplate(
            input_variables=["job_title", "experience", "skills"],
            template="""
            Write a clean and visually formatted HTML resume for a {job_title} position.
            Include a Summary section using the following experience:
            {experience}
            
            Also include a Skills section using the following:
            {skills}
            
            Create a professional layout using HTML tags like <h2>, <ul>, <li>, <b>.
            Keep it readable and minimal, use sections:
            - Name and Contact Info
            - Summary
            - Skills
            - Experience
            Do not include <html>, <head>, or <body> tags.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(job_title=job_title, experience=experience, skills=skills)
        st.markdown("### ✅ Formatted Resume Preview")
        st.components.v1.html(output, height=600, scrolling=True)

# Cover Letter Generator
elif menu == "Cover Letter Generator":
    st.subheader("📬 Cover Letter Generator")
    job_title = st.text_input("🔹 Job Title")
    company = st.text_input("🔹 Company Name")
    motivation = st.text_area("🔹 Why this role?")
    achievements = st.text_area("🔹 Key Achievements")

    if st.button("✍️ Generate Cover Letter"):
        prompt = PromptTemplate(
            input_variables=["job_title", "company", "motivation", "achievements"],
            template="""
            Write a formal cover letter for a {job_title} position at {company}.
            Reason for interest: {motivation}
            Highlight these achievements: {achievements}
            Use a polite, confident tone.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(
            job_title=job_title, company=company,
            motivation=motivation, achievements=achievements
        )
        st.markdown("### ✅ Generated Cover Letter")
        st.code(output, language='markdown')

# CV Analyzer
elif menu == "CV Analyzer":
    st.subheader("📊 CV Analyzer")
    uploaded = st.file_uploader("📎 Upload your resume (PDF only)", type=['pdf'])

    if uploaded:
        reader = PdfReader(uploaded)
        text = "".join([page.extract_text() for page in reader.pages])
        st.text_area("📄 Extracted Resume Text", text, height=250)

        if st.button("📌 Analyze Resume"):
            prompt = PromptTemplate(
                input_variables=["cv_text"],
                template="""
                Analyze the following resume and provide suggestions for improvement.
                Be specific about missing sections, poor formatting, or vague statements.

                Resume:
                {cv_text}
                """
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            feedback = chain.run(cv_text=text)
            st.markdown("### 💡 Suggestions:")
            st.write(feedback)

# LinkedIn Summary Generator
elif menu == "LinkedIn Summary Generator":
    st.subheader("🔗 LinkedIn Summary Generator")
    name = st.text_input("🔹 Your Name")
    profession = st.text_input("🔹 Your Profession")
    goals = st.text_area("🔹 Career Goals")
    key_skills = st.text_area("🔹 Key Skills")

    if st.button("✍️ Generate LinkedIn Summary"):
        prompt = PromptTemplate(
            input_variables=["name", "profession", "goals", "key_skills"],
            template="""
            Write a compelling LinkedIn summary for {name}, a {profession}.
            Focus on: {goals}. Mention skills like {key_skills}.
            Use a friendly, engaging tone.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(name=name, profession=profession, goals=goals, key_skills=key_skills)
        st.markdown("### ✅ LinkedIn Summary")
        st.code(summary)

# Mock Interview Chatbot
elif menu == "Mock Interview Chatbot":
    st.subheader("🎙️ Mock Interview Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💬 You (Candidate):", key="interview_input")

    if st.button("📤 Send") and user_input:
        conversation = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        prompt = PromptTemplate(
            input_variables=["conversation", "user_input"],
            template="""
            You're simulating a mock interview. Continue the following interview. Ask one new question 
            or give constructive feedback.

            Interview so far:
            {conversation}

            Candidate's latest response:
            {user_input}
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        reply = chain.run(conversation=conversation, user_input=user_input)
        st.session_state.chat_history.append((user_input, reply))

    for question, reply in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {question}")
        st.markdown(f"**🧠 Coach:** {reply}")
