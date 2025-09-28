import os
import streamlit as st
import time
from typing import List
from dotenv import load_dotenv

# To remove the system has reached its maximum watch limit
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Load environment variables
load_dotenv()

# Import your existing bot class
try:
    from bot import SimpleContentCreator
except ImportError:
    st.error("Could not import SimpleContentCreator from bot.py. Make sure bot.py is in the same directory.")
    st.stop()

# Hardcoded API key - replace with your actual key
GROQ_API_KEY = "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"

def get_api_key():
    # Return the hardcoded API key
    return GROQ_API_KEY

def extract_file_content(uploaded_file):
    """Extract content from uploaded file"""
    try:
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
                st.error("Install PyPDF2 to process PDF files: pip install PyPDF2")
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
                st.error("Install python-docx to process Word files: pip install python-docx")
                return None
            except Exception as e:
                st.error(f"Error reading Word document: {str(e)}")
                return None
        else:
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                st.error(f"Cannot decode file {uploaded_file.name}. Please ensure it's a text-based file.")
                return None
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def process_project_data(creator, project_name, urls_list, uploaded_files, additional_text):
    """Process all project data and setup the creator"""
    try:
        # Prepare combined whitepaper text from files and additional text
        combined_text_content = ""
      
        # Add additional text if provided
        if additional_text and additional_text.strip():
            combined_text_content += additional_text + "\n\n"
      
        # Process uploaded files and add to combined text
        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = extract_file_content(uploaded_file)
                if content:
                    combined_text_content += f"\n--- Content from {uploaded_file.name} ---\n\n{content}\n\n"
      
        # Call the original setup_project method
        creator.setup_project(
            project_name=project_name,
            urls=urls_list,
            whitepaper_path=None  # We're not using file paths, just text content
        )
      
        # If we have combined text content, we need to add it to the vector store
        if combined_text_content.strip():
            # Create a document from combined text and add to existing vector store
            from langchain_core.documents import Document
          
            if not creator.vector_store:
                # If no vector store exists yet (no URLs processed), create one with text content
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
                # Add to existing vector store
                text_doc = Document(page_content=combined_text_content)
                chunks = creator.text_splitter.split_documents([text_doc])
                if chunks:
                    creator.vector_store.add_documents(chunks)
      
        # Return the number of documents in the vector store
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
  
    # Get API key
    groq_api_key = get_api_key()
    
    if not groq_api_key:
        st.error("❌ GROQ API key not found!")
        st.info("Please set your GROQ_API_KEY in:")
        st.code("1. Environment variables, or\n2. Streamlit secrets (for deployment)")
        st.stop()
  
    # Initialize session state
    if 'creator' not in st.session_state:
        try:
            st.session_state.creator = SimpleContentCreator(groq_api_key)
        except Exception as e:
            st.error(f"Error initializing content creator: {str(e)}")
            st.error("Please check your API key and ensure all dependencies are installed.")
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
            placeholder="https://example.com/docs\nhttps://example.com/whitepaper",
            help="Enter documentation URLs, one per line"
        )
      
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
                file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 'Unknown size'
                st.write(f"• {file.name} ({file_size} bytes)")
      
        # Additional Text
        st.subheader("📝 Paste Content")
        additional_text = st.text_area(
            "Paste any additional content", 
            height=100, 
            placeholder="Paste whitepaper, documentation, or any other text here...",
            help="Any additional text content to include"
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
                    with st.spinner("Processing your data... This may take a few minutes."):
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
                                st.success(f"✅ Data processed successfully!")
                                st.info(f"📊 Vector store contains {doc_count} document chunks")
                            else:
                                st.error("❌ No data could be processed. Please check your sources and try again.")
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
  
    # Main content area
    if not st.session_state.data_uploaded:
        st.info("👈 Please upload your data first using the sidebar")
      
        # Instructions
        col1, col2 = st.columns(2)
      
        with col1:
            st.markdown("""
            ### 🚀 How it works:
            1. **Add your project name** - Give your project a name
            2. **Upload your data** - Add URLs, files, or paste content
            3. **Process the data** - Click the process button
            4. **Create content** - Write what you want and choose the format
            """)
          
        with col2:
            st.markdown("""
            ### 📁 Supported formats:
            - 📄 **PDF files** - Whitepapers, research papers
            - 📝 **Word documents** - Documentation, reports  
            - 🔗 **Website URLs** - Documentation sites, blogs
            - ✏️ **Text content** - Any text you want to include
            """)
      
        # Example section
        with st.expander("📋 Example Usage"):
            st.markdown("""
            **URLs Example:**
            ```
            https://docs.solana.com/introduction
            https://docs.solana.com/consensus
            https://ethereum.org/en/developers/docs/
            ```
          
            **Content Ideas:**
            - "Explain how the consensus mechanism works"
            - "Write about the tokenomics and incentive structure"
            - "Compare our approach to traditional solutions"
            - "Deep dive into the technical architecture"
            """)
      
    else:
        # Content creation interface
        st.header(f"✨ Create Content for {st.session_state.project_name}")
      
        # User prompt input
        user_prompt = st.text_area(
            "What do you want to write about?", 
            height=100,
            placeholder="e.g., 'Explain how the consensus mechanism works' or 'Write about the tokenomics and incentive structure'",
            help="Describe what you want to create content about in plain language"
        )
      
        # Content type and options
        col1, col2, col3 = st.columns([2, 2, 1])
      
        with col1:
            content_type = st.selectbox(
                "Content Type", 
                ["Twitter Thread", "Blog Post"],
                help="Choose the format for your content"
            )
      
        with col2:
            if content_type == "Twitter Thread":
                thread_length = st.number_input(
                    "Number of tweets", 
                    min_value=1, 
                    max_value=20, 
                    value=6,
                    help="How many tweets in the thread"
                )
            else:
                blog_length = st.selectbox(
                    "Blog length", 
                    ["short", "medium", "long"],
                    index=1,
                    help="Short: 800-1200 words, Medium: 1500-2500 words, Long: 2500-4000 words"
                )
      
        with col3:
            st.write("")  # Spacer
            generate_button = st.button("🎯 Generate", type="primary")
      
        # Generate content
        if generate_button:
            if not user_prompt.strip():
                st.error("Please enter what you want to write about!")
            else:
                with st.spinner("🤖 Creating your content... This may take 1-2 minutes."):
                    try:
                        if content_type == "Twitter Thread":
                            content = st.session_state.creator.create_twitter_thread(
                                user_prompt, 
                                thread_length
                            )
                        else:
                            content = st.session_state.creator.create_blog_post(
                                user_prompt, 
                                blog_length
                            )
                      
                        if content and not content.startswith("Error:"):
                            st.success("✅ Content generated successfully!")
                          
                            # Display the content
                            st.subheader("Your Generated Content")
                            st.text_area(
                                "Generated Content Output",
                                content, 
                                height=500, 
                                key="generated_content",
                                help="Copy this content or use the download button below",
                                label_visibility="collapsed"
                            )
                          
                            # Download button
                            file_extension = "txt" if content_type == "Twitter Thread" else "md"
                            filename = f"{user_prompt[:30].replace(' ', '_')}_{content_type.lower().replace(' ', '_')}.{file_extension}"
                          
                            st.download_button(
                                label=f"📥 Download {content_type}",
                                data=content,
                                file_name=filename,
                                mime="text/plain",
                                help="Download your generated content as a file"
                            )
                          
                            # Option to create more content
                            if st.button("🔄 Create Another", key="create_another"):
                                st.rerun()
                              
                        else:
                            st.error("❌ Content generation failed. Please try again or check your data sources.")
                            if content:
                                st.error(content)
                          
                    except Exception as e:
                        st.error(f"❌ Error generating content: {str(e)}")
                        st.error("Please check your API key and internet connection.")
        
        # Reset option
        st.divider()
        if st.button("🔄 Start New Project", type="secondary"):
            # Clear session state
            st.session_state.data_uploaded = False
            st.session_state.project_name = ""
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")