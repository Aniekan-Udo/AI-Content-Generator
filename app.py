import streamlit as st
import os
import time
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.docstore.document import Document

# Configure your API key here
GROQ_API_KEY = "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"

class SimpleContentCreator:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.project_name = None
        self.vector_store = None
        
    def setup_project(self, project_name: str, urls: List[str], uploaded_files=None, additional_text: str = None):
        """Setup project by loading documents from URLs, files, and text"""
        self.project_name = project_name
        all_documents = []
        
        # Process URLs
        for url in urls:
            if url.strip():
                try:
                    loader = WebBaseLoader([url])
                    docs = loader.load()
                    chunks = self.text_splitter.split_documents(docs)
                    all_documents.extend(chunks)
                except Exception as e:
                    st.error(f"Error loading {url}: {str(e)}")
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    content = self._extract_file_content(uploaded_file)
                    if content:
                        doc = Document(page_content=content, metadata={"source": uploaded_file.name})
                        chunks = self.text_splitter.split_documents([doc])
                        all_documents.extend(chunks)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Process additional text
        if additional_text and additional_text.strip():
            doc = Document(page_content=additional_text)
            chunks = self.text_splitter.split_documents([doc])
            all_documents.extend(chunks)
        
        # Create vector store
        if all_documents:
            self.vector_store = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                collection_name=f"{project_name.lower()}_{int(time.time())}"
            )
            return len(all_documents)
        return 0
    
    def _extract_file_content(self, uploaded_file):
        """Extract content from uploaded file"""
        file_content = uploaded_file.read()
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.txt') or file_name.endswith('.md'):
            return file_content.decode('utf-8')
        elif file_name.endswith('.pdf'):
            try:
                import PyPDF2
                from io import BytesIO
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                return "\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                st.error("Install PyPDF2 to process PDF files")
                return None
        elif file_name.endswith(('.docx', '.doc')):
            try:
                from docx import Document as DocxDocument
                from io import BytesIO
                doc = DocxDocument(BytesIO(file_content))
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                st.error("Install python-docx to process Word files")
                return None
        else:
            try:
                return file_content.decode('utf-8')
            except:
                st.error(f"Cannot process file: {uploaded_file.name}")
                return None
    
    def create_twitter_thread(self, topic: str, thread_length: int = 6) -> str:
        """Create Twitter thread"""
        if not self.vector_store:
            return "Error: No project setup."
        
        docs = self.vector_store.similarity_search(topic, k=25)
        research_content = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
CRITICAL CREATIVITY MANDATE: Create completely ORIGINAL Twitter thread about "{topic}" for {self.project_name}.

FORBIDDEN: Generic definitions, obvious benefits, predictable structures
REQUIRED: Contrarian insights, hidden mechanics, counterintuitive truths

Create exactly {thread_length} tweets using this research:
{research_content}

Format as: 1/{thread_length}: [content] etc.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def create_blog_post(self, topic: str, length: str = "medium") -> str:
        """Create blog post"""
        if not self.vector_store:
            return "Error: No project setup."
        
        docs = self.vector_store.similarity_search(topic, k=35)
        research_content = "\n".join([doc.page_content for doc in docs])
        
        length_guide = {
            "short": "800-1200 words",
            "medium": "1500-2500 words", 
            "long": "2500-4000 words"
        }
        
        prompt = f"""
Create a GROUNDBREAKING blog post about "{topic}" for {self.project_name}.
Target: {length_guide.get(length, "1500-2500 words")}

FORBIDDEN: Standard introductions, obvious explanations, predictable structures
REQUIRED: Original thesis, contrarian perspectives, paradigm shifts

Research: {research_content}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

def main():
    st.set_page_config(
        page_title="AI Content Creator",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AI Content Creator")
    st.markdown("Upload your data and create amazing content!")
    
    # Initialize the creator with pre-configured API key
    if 'creator' not in st.session_state:
        st.session_state.creator = SimpleContentCreator(GROQ_API_KEY)
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Upload Your Data")
        
        project_name = st.text_input("Project Name", placeholder="Enter your project name")
        
        # URLs
        st.subheader("🔗 Website URLs")
        urls_text = st.text_area("Add URLs (one per line)", height=100, placeholder="https://example.com/docs\nhttps://example.com/whitepaper")
        
        # File Upload
        st.subheader("📄 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md'],
            help="Upload PDFs, Word docs, text files, or markdown files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files ready to upload")
            for file in uploaded_files:
                st.write(f"• {file.name}")
        
        # Additional Text
        st.subheader("📝 Paste Content")
        additional_text = st.text_area("Paste any additional content", height=100, placeholder="Paste whitepaper, documentation, or any other text here...")
        
        # Upload Button
        if st.button("🚀 Process Data", type="primary"):
            if not project_name:
                st.error("Please enter a project name!")
            else:
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                
                if not urls and not uploaded_files and not additional_text.strip():
                    st.error("Please add at least one data source!")
                else:
                    with st.spinner("Processing your data... This may take a few minutes."):
                        doc_count = st.session_state.creator.setup_project(
                            project_name, urls, uploaded_files, additional_text
                        )
                        
                        if doc_count > 0:
                            st.session_state.data_uploaded = True
                            st.success(f"✅ Data processed! {doc_count} chunks loaded")
                        else:
                            st.error("❌ No data could be processed")
    
    # Main content area
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload your data first using the sidebar")
        
        st.markdown("""
        ### How it works:
        1. **Add your project name** - Give your project a name
        2. **Upload your data** - Add URLs, files, or paste content
        3. **Process the data** - Click the process button
        4. **Create content** - Write what you want and choose the format
        
        ### Supported formats:
        - 📄 **PDF files** - Whitepapers, research papers
        - 📝 **Word documents** - Documentation, reports  
        - 🔗 **Website URLs** - Documentation sites, blogs
        - ✏️ **Text content** - Any text you want to include
        """)
        
    else:
        # Content creation interface
        st.header("✨ Create Your Content")
        
        # User prompt input
        user_prompt = st.text_area(
            "What do you want to write about?", 
            height=100,
            placeholder="e.g., 'Explain how the consensus mechanism works' or 'Write about the tokenomics and incentive structure'"
        )
        
        # Content type selection
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            content_type = st.selectbox("Content Type", ["Twitter Thread", "Blog Post"])
        
        with col2:
            if content_type == "Twitter Thread":
                thread_length = st.number_input("Number of tweets", min_value=3, max_value=15, value=6)
            else:
                blog_length = st.selectbox("Blog length", ["short", "medium", "long"])
        
        with col3:
            st.write("")  # Spacer
            generate_button = st.button("🎯 Generate", type="primary")
        
        # Generate content
        if generate_button:
            if not user_prompt.strip():
                st.error("Please enter what you want to write about!")
            else:
                with st.spinner("Creating your content... This may take 1-2 minutes."):
                    try:
                        if content_type == "Twitter Thread":
                            content = st.session_state.creator.create_twitter_thread(user_prompt, thread_length)
                        else:
                            content = st.session_state.creator.create_blog_post(user_prompt, blog_length)
                        
                        st.success("✅ Content generated!")
                        
                        # Display the content
                        st.subheader("Your Generated Content")
                        st.text_area("", content, height=500, key="generated_content")
                        
                        # Download button
                        file_extension = "txt" if content_type == "Twitter Thread" else "md"
                        filename = f"{user_prompt[:30].replace(' ', '_')}_{content_type.lower().replace(' ', '_')}.{file_extension}"
                        
                        st.download_button(
                            label=f"📥 Download {content_type}",
                            data=content,
                            file_name=filename,
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating content: {str(e)}")

if __name__ == "__main__":
    main()