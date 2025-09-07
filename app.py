#!/usr/bin/env python3
"""
Interview Question Forecaster - Streamlit Web App

Analyzes a job description and resume to generate likely interview questions
(behavioral + technical) and auto-drafts 60-second STAR answers in your voice.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from forecaster import InterviewForecaster, AnalysisResult, InterviewQuestion


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Interview Question Forecaster",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .question-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .answer-text {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-top: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Interview Question Forecaster</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Analyze your resume and job description to generate likely interview questions<br>
        and auto-draft 60-second STAR answers in your voice
    </div>
    """, unsafe_allow_html=True)
    
    # Get API key from environment
    api_key = os.environ.get("General")
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please set the 'General' environment variable.")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success("✅ OpenAI API key loaded from environment")
        
        # Model selection
        model = st.selectbox(
            "AI Model",
            ["gpt-5-mini", "gpt-4.1-mini"],
            index=0,
            help="Choose the OpenAI model to use for analysis"
        )
        
        # Temperature setting
        temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📄 Upload Your Resume</div>', unsafe_allow_html=True)
        
        resume_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'txt', 'md'],
            help="Upload your resume in PDF, TXT, or MD format"
        )
        
        resume_text = st.text_area(
            "Or paste your resume text here",
            height=200,
            help="Alternative to file upload - paste your resume content directly"
        )
    
    with col2:
        st.markdown('<div class="sub-header">💼 Upload Job Description</div>', unsafe_allow_html=True)
        
        jd_file = st.file_uploader(
            "Choose a job description file",
            type=['pdf', 'txt', 'md'],
            help="Upload the job description in PDF, TXT, or MD format"
        )
        
        jd_text = st.text_area(
            "Or paste job description text here",
            height=200,
            help="Alternative to file upload - paste the job description directly"
        )
    
    # Process button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Generate Interview Questions",
            type="primary",
            use_container_width=True
        )
    
    # Processing logic
    if process_button:
        # Validate inputs
        if not resume_file and not resume_text.strip():
            st.error("❌ Please upload a resume file or paste resume text.")
            return
        
        if not jd_file and not jd_text.strip():
            st.error("❌ Please upload a job description file or paste job description text.")
            return
        
        # Process files
        try:
            with st.spinner("🔄 Processing your documents and generating questions..."):
                # Initialize forecaster
                forecaster = InterviewForecaster(api_key=api_key)
                
                # Handle resume
                if resume_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(resume_file.getvalue())
                        resume_content = forecaster._extract_text_from_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                else:
                    resume_content = resume_text.strip()
                
                # Handle job description
                if jd_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(jd_file.getvalue())
                        jd_content = forecaster._extract_text_from_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                else:
                    jd_content = jd_text.strip()
                
                # Generate questions
                result = forecaster.question_generator.generate_questions(resume_content, jd_content)
                
                # Store result in session state
                st.session_state['analysis_result'] = result
                st.session_state['forecaster'] = forecaster
        
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        forecaster = st.session_state['forecaster']
        
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Summary section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions Generated", len(result.questions))
        
        with col2:
            technical_count = sum(1 for q in result.questions if q.category == 'technical')
            st.metric("Technical Questions", technical_count)
        
        with col3:
            behavioral_count = sum(1 for q in result.questions if q.category == 'behavioral')
            st.metric("Behavioral Questions", behavioral_count)
        
        # Summary
        if result.summary:
            st.markdown("### 📝 Summary")
            st.info(result.summary)
        
        # Key skills and gaps
        col1, col2 = st.columns(2)
        
        with col1:
            if result.key_skills:
                st.markdown("### 🎯 Key Skills to Highlight")
                for skill in result.key_skills:
                    st.markdown(f"• {skill}")
        
        with col2:
            if result.experience_gaps:
                st.markdown("### ⚠️ Potential Experience Gaps")
                for gap in result.experience_gaps:
                    st.markdown(f"• {gap}")
        
        # Questions and answers
        st.markdown("### 🎤 Interview Questions & STAR Responses")
        
        for i, question in enumerate(result.questions, 1):
            # Determine confidence badge class
            if question.confidence >= 0.8:
                confidence_class = "confidence-high"
            elif question.confidence >= 0.6:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            # Question card
            st.markdown(f"""
            <div class="question-card">
                <h4>Q{i}: {question.question}</h4>
                <span class="confidence-badge {confidence_class}">
                    {question.category.title()} • {question.confidence:.1%} confidence
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer
            st.markdown(f"""
            <div class="answer-text">
                <strong>STAR Response:</strong><br>
                {question.answer}
            </div>
            """, unsafe_allow_html=True)
        
        # Download PDF button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Download PDF Crib Sheet", use_container_width=True):
                try:
                    with st.spinner("Generating PDF..."):
                        # Create temporary PDF file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            forecaster.pdf_exporter.export_crib_sheet(result, tmp_file.name)
                            
                            # Read the PDF file
                            with open(tmp_file.name, 'rb') as pdf_file:
                                pdf_data = pdf_file.read()
                            
                            # Clean up
                            os.unlink(tmp_file.name)
                        
                        # Provide download button
                        st.download_button(
                            label="📥 Download Your Interview Crib Sheet",
                            data=pdf_data,
                            file_name="interview_crib_sheet.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")


if __name__ == "__main__":
    main()
