import os
import streamlit as st
from dotenv import load_dotenv
import time
from typing import List
import PyPDF2
from io import BytesIO
load_dotenv()

import os
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "ai-content-generator/0.1")

# To remove the system has reached its maximum watch limit
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Hardcoded API key
GROQ_API_KEY = st.secrets["API_KEY"]
if not GROQ_API_KEY:
    raise ValueError("API_KEY environment variable not set.")

# Import your existing bot class
try:
    from bot import SimpleContentCreator
    print("Successfully imported SimpleContentCreator")
except ImportError as e:
    st.error(f"Could not import SimpleContentCreator from bot.py: {e}")
    st.stop()

def extract_file_content(uploaded_file):
    """Extract content from uploaded file"""
    try:
        file_content = uploaded_file.read()
        file_name = uploaded_file.name.lower()
      
        if file_name.endswith('.txt') or file_name.endswith('.md'):
            return file_content.decode('utf-8')
        elif file_name.endswith('.pdf'):
            try:
                
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                return "\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                st.error("PyPDF2 not available")
                return None
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return None
        elif file_name.endswith(('.docx', '.doc')):
            try:
                from docx import Document as DocxDocument
                from io import BytesIO
                doc = DocxDocument(BytesIO(file_content))
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                st.error("python-docx not available")
                return None
            except Exception as e:
                st.error(f"Error reading Word document: {str(e)}")
                return None
        else:
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                st.error(f"Cannot decode file {uploaded_file.name}")
                return None
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def process_project_data(creator, project_name, urls_list, uploaded_files, additional_text):
    """Process all project data and setup the creator"""
    try:
        combined_text_content = ""
      
        if additional_text and additional_text.strip():
            combined_text_content += additional_text + "\n\n"
      
        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = extract_file_content(uploaded_file)
                if content:
                    combined_text_content += f"\n--- Content from {uploaded_file.name} ---\n\n{content}\n\n"
      
        creator.setup_project(
            project_name=project_name,
            urls=urls_list,
            whitepaper_path=None
        )
      
        if combined_text_content.strip():
            from langchain_core.documents import Document
          
            if not creator.vector_store:
                text_doc = Document(page_content=combined_text_content)
                chunks = creator.text_splitter.split_documents([text_doc])
              
                if chunks:
                    from langchain_community.vectorstores import Chroma
                    creator.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=creator.embeddings,
                        collection_name=f"{project_name.lower()}_{int(time.time())}"
                    )
            else:
                text_doc = Document(page_content=combined_text_content)
                chunks = creator.text_splitter.split_documents([text_doc])
                if chunks:
                    creator.vector_store.add_documents(chunks)
      
        if creator.vector_store:
            return len(creator.vector_store.get()['ids'])
        return 0
      
    except Exception as e:
        st.error(f"Error setting up project: {str(e)}")
        return 0

def main():
    st.set_page_config(
        page_title="AI Content Creator",
        page_icon="🚀",
        layout="wide"
    )
  
    st.title("🚀 AI Content Creator")
    st.markdown("Upload your data and create amazing content!")
  
    # Initialize session state
    if 'creator' not in st.session_state:
        try:
            st.session_state.creator = SimpleContentCreator(GROQ_API_KEY)
        except Exception as e:
            st.error(f"Error initializing content creator: {str(e)}")
            st.stop()
          
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'project_name' not in st.session_state:
        st.session_state.project_name = ""
  
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Upload Your Data")
      
        project_name = st.text_input(
            "Project Name", 
            value=st.session_state.project_name,
            placeholder="Enter your project name"
        )
      
        # URLs
        st.subheader("🔗 Website URLs")
        urls_text = st.text_area(
            "Add URLs (one per line)", 
            height=100, 
            placeholder="https://example.com/docs\nhttps://example.com/whitepaper"
        )
      
        # File Upload
        st.subheader("📄 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md']
        )
      
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files ready")
            for file in uploaded_files:
                st.write(f"• {file.name}")
      
        # Additional Text
        st.subheader("📝 Paste Content")
        additional_text = st.text_area(
            "Paste any additional content", 
            height=100, 
            placeholder="Paste content here..."
        )
      
        # Upload Button
        if st.button("🚀 Process Data", type="primary"):
            if not project_name.strip():
                st.error("Please enter a project name!")
            else:
                urls_list = [url.strip() for url in urls_text.split('\n') if url.strip()]
              
                if not urls_list and not uploaded_files and not additional_text.strip():
                    st.error("Please add at least one data source!")
                else:
                    with st.spinner("Processing data..."):
                        try:
                            doc_count = process_project_data(
                                st.session_state.creator, 
                                project_name, 
                                urls_list, 
                                uploaded_files, 
                                additional_text
                            )
                          
                            if doc_count > 0:
                                st.session_state.data_uploaded = True
                                st.session_state.project_name = project_name
                                st.success(f"✅ Data processed! {doc_count} chunks")
                            else:
                                st.error("❌ No data could be processed")
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
  
    # Main content area
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload your data first using the sidebar")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### How it works:
            1. Add your project name
            2. Upload your data
            3. Process the data
            4. Create content
            """)
          
        with col2:
            st.markdown("""
            ### Supported formats:
            - PDF files
            - Word documents  
            - Website URLs
            - Text content
            """)
      
    else:
        # Content creation interface
        st.header(f"✨ Create Content for {st.session_state.project_name}")
      
        user_prompt = st.text_area(
            "What do you want to write about?", 
            height=100,
            placeholder="e.g., 'Explain how the consensus mechanism works'"
        )
      
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            content_type = st.selectbox("Content Type", ["Twitter Thread", "Blog Post"])

        with col2:
            if content_type == "Blog Post":  # Only show options for blog posts now
                blog_length = st.selectbox("Blog length", ["short", "medium", "long"], index=1)
            else:
                st.write("")  # Empty space to keep layout consistent

        with col3:
            st.write("")
            generate_button = st.button("🎯 Generate", type="primary")
            
        if generate_button:
            if not user_prompt.strip():
                st.error("Please enter what you want to write about!")
            else:
                with st.spinner("Creating content..."):
                    try:
                        if content_type == "Twitter Thread":
                            content = st.session_state.creator.create_twitter_thread(user_prompt)
                        else:
                            content = st.session_state.creator.create_blog_post(user_prompt, blog_length)
                      
                        if content and not content.startswith("Error:"):
                            st.success("✅ Content generated!")
                            st.text_area("Generated Content", content, height=500)
                            
                            file_extension = "txt" if content_type == "Twitter Thread" else "md"
                            filename = f"{user_prompt[:30].replace(' ', '_')}.{file_extension}"
                          
                            st.download_button(
                                label=f"📥 Download {content_type}",
                                data=content,
                                file_name=filename,
                                mime="text/plain"
                            )
                        else:
                            st.error("❌ Content generation failed")
                            if content:
                                st.error(content)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        if st.button("🔄 Start New Project"):
            st.session_state.data_uploaded = False
            st.session_state.project_name = ""
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")
