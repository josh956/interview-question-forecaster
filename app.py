#!/usr/bin/env python3
"""
Interview Question Forecaster - Streamlit Web App

Analyzes a job description and resume to generate likely interview questions
(behavioral + technical) and auto-drafts 60-second STAR answers in your voice.
"""

import streamlit as st
import tempfile
import os
import json
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import openai
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER


@dataclass
class InterviewQuestion:
    """Represents a single interview question with its STAR response."""
    question: str
    answer: str
    category: str  # 'behavioral' or 'technical'
    confidence: float  # 0.0 to 1.0


@dataclass
class AnalysisResult:
    """Contains the complete analysis results."""
    questions: List[InterviewQuestion]
    summary: str
    key_skills: List[str]
    experience_gaps: List[str]


class PDFProcessor:
    """Handles PDF text extraction."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF {pdf_path}: {str(e)}")


class OpenAIQuestionGenerator:
    """Generates interview questions using OpenAI API."""
    
    def __init__(self, api_key: str):
        """Initialize the OpenAI client."""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-5-mini"
    
    def generate_questions(self, resume_text: str, job_description: str, num_questions: int = 10) -> AnalysisResult:
        """Generate interview questions and STAR responses."""
        
        prompt = self._create_analysis_prompt(resume_text, job_description, num_questions)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert interview coach and career advisor. You analyze resumes and job descriptions to predict likely interview questions and craft compelling STAR responses."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            raise RuntimeError(f"Error generating questions with OpenAI: {str(e)}")
    
    def _create_analysis_prompt(self, resume_text: str, job_description: str, num_questions: int = 10) -> str:
        """Create the analysis prompt for OpenAI."""
        return f"""
Analyze the following job description and candidate resume to generate {num_questions} likely interview questions with STAR responses.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Please provide your analysis in the following JSON format:

{{
    "summary": "Brief 2-3 sentence summary of the candidate's fit for this role",
    "key_skills": ["skill1", "skill2", "skill3"],
    "experience_gaps": ["gap1", "gap2"],
    "questions": [
        {{
            "question": "What is your experience with [specific technology/skill]?",
            "answer": "Situation: [Context] Task: [What needed to be done] Action: [What you did] Result: [Outcome]",
            "category": "technical",
            "confidence": 0.9
        }},
        {{
            "question": "Tell me about a time when you had to [behavioral scenario]",
            "answer": "Situation: [Context] Task: [What needed to be done] Action: [What you did] Result: [Outcome]",
            "category": "behavioral", 
            "confidence": 0.8
        }}
    ]
}}

Requirements:
1. Generate exactly {num_questions} questions (mix of behavioral and technical)
2. Each answer should be a complete STAR response (60-90 seconds when spoken)
3. Answers should be written in the candidate's voice based on their resume
4. Technical questions should focus on skills mentioned in the JD
5. Behavioral questions should relate to the role's requirements
6. Confidence scores should reflect how likely the question is to be asked
7. Include specific technologies, frameworks, and scenarios from the JD
8. Make answers authentic and specific to the candidate's experience

Focus on:
- Skills alignment between resume and JD
- Industry-specific questions
- Leadership and teamwork scenarios
- Problem-solving examples
- Technical depth questions
- Cultural fit questions
"""
    
    def _parse_response(self, content: str) -> AnalysisResult:
        """Parse the OpenAI response into structured data."""
        try:
            # Extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in response")
            
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            questions = []
            for q_data in data.get('questions', []):
                question = InterviewQuestion(
                    question=q_data['question'],
                    answer=q_data['answer'],
                    category=q_data['category'],
                    confidence=q_data['confidence']
                )
                questions.append(question)
            
            return AnalysisResult(
                questions=questions,
                summary=data.get('summary', ''),
                key_skills=data.get('key_skills', []),
                experience_gaps=data.get('experience_gaps', [])
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing OpenAI response as JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing OpenAI response: {str(e)}")


class PDFExporter:
    """Handles PDF crib sheet generation."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='QuestionStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            textColor='#2c3e50'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AnswerStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            leftIndent=20,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='HeaderStyle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor='#34495e'
        ))
    
    def export_crib_sheet(self, result: AnalysisResult, output_path: str):
        """Export the analysis results to a PDF crib sheet."""
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Title
        story.append(Paragraph("Interview Question Crib Sheet", self.styles['HeaderStyle']))
        story.append(Spacer(1, 12))
        
        # Summary
        if result.summary:
            story.append(Paragraph("Summary", self.styles['QuestionStyle']))
            story.append(Paragraph(result.summary, self.styles['AnswerStyle']))
            story.append(Spacer(1, 12))
        
        # Key Skills
        if result.key_skills:
            story.append(Paragraph("Key Skills to Highlight", self.styles['QuestionStyle']))
            skills_text = " • ".join(result.key_skills)
            story.append(Paragraph(skills_text, self.styles['AnswerStyle']))
            story.append(Spacer(1, 12))
        
        # Experience Gaps
        if result.experience_gaps:
            story.append(Paragraph("Potential Experience Gaps", self.styles['QuestionStyle']))
            gaps_text = " • ".join(result.experience_gaps)
            story.append(Paragraph(gaps_text, self.styles['AnswerStyle']))
            story.append(Spacer(1, 12))
        
        # Questions and Answers
        story.append(Paragraph("Interview Questions & STAR Responses", self.styles['QuestionStyle']))
        story.append(Spacer(1, 6))
        
        for i, q in enumerate(result.questions, 1):
            # Question
            question_text = f"Q{i}: {q.question}"
            story.append(Paragraph(question_text, self.styles['QuestionStyle']))
            
            # Category and confidence
            category_text = f"[{q.category.title()}] Confidence: {q.confidence:.1%}"
            story.append(Paragraph(category_text, self.styles['AnswerStyle']))
            
            # Answer
            story.append(Paragraph(q.answer, self.styles['AnswerStyle']))
            
            # Add page break if we're at question 5 to keep it to 1 page
            if i == 5:
                story.append(PageBreak())
        
        doc.build(story)


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file (PDF or text)."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() == '.pdf':
        return PDFProcessor.extract_text_from_pdf(file_path)
    elif path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: .pdf, .txt, .md")


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
            ["gpt-5-mini"],
            index=0,
            help="Choose the OpenAI model to use for analysis"
        )
        
        # Short mode toggle
        short_mode = st.toggle(
            "📝 Short Mode",
            value=False,
            help="Generate 3 questions instead of 10 for quick preparation"
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
                # Initialize components
                question_generator = OpenAIQuestionGenerator(api_key)
                pdf_exporter = PDFExporter()
                
                # Handle resume
                if resume_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(resume_file.getvalue())
                        resume_content = extract_text_from_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                else:
                    resume_content = resume_text.strip()
                
                # Handle job description
                if jd_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(jd_file.getvalue())
                        jd_content = extract_text_from_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                else:
                    jd_content = jd_text.strip()
                
                # Generate questions
                num_questions = 3 if short_mode else 10
                result = question_generator.generate_questions(resume_content, jd_content, num_questions)
                
                # Store result in session state
                st.session_state['analysis_result'] = result
        
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        pdf_exporter = PDFExporter()  # Create new instance instead of storing in session state
        
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
                            pdf_exporter.export_crib_sheet(result, tmp_file.name)
                            
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