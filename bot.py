import os
import time
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleContentCreator:
    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("GROQ API key is required")
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.project_name = None
        self.vector_store = None
      
    def setup_project(self, project_name: str, urls: List[str], whitepaper_path: str = None):
        """Setup project by loading documents from specified URLs and whitepaper"""
        print(f"Setting up project: {project_name}")
        print(f"Processing {len(urls)} URLs...")
      
        self.project_name = project_name
        all_documents = []
      
        # Process each URL directly (no automatic subpage discovery)
        for url in urls:
            print(f"Processing: {url}")
            try:
                loader = WebBaseLoader([url])
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                all_documents.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from {url}")
            except Exception as e:
                print(f"  Error loading {url}: {str(e)}")
          
        # Process whitepaper if provided
        if whitepaper_path and os.path.exists(whitepaper_path):
            print(f"Processing whitepaper: {whitepaper_path}")
            try:
                loader = TextLoader(whitepaper_path)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                all_documents.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from whitepaper")
            except Exception as e:
                print(f"  Error loading whitepaper: {str(e)}")
          
        # Create vector store with all content
        if all_documents:
            print(f"Creating knowledge base with {len(all_documents)} document chunks")
            self.vector_store = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                collection_name=f"{project_name.lower()}_{int(time.time())}"
            )
            print(f"Setup complete for {project_name}")
        else:
            print("Warning: No documents were loaded")
  
    def create_twitter_thread(self, topic: str, thread_length: int = 6) -> str:
        """Create Twitter thread using comprehensive research"""
        if not self.vector_store:
            return "Error: No project setup. Run setup_project() first."
          
        print(f"Creating Twitter thread about: {topic}")
      
        # Get comprehensive research from ALL documents
        docs = self.vector_store.similarity_search(topic, k=25)
        research_content = "\n".join([doc.page_content for doc in docs])
      
        prompt = f"""
        CRITICAL CREATIVITY MANDATE: You MUST create completely ORIGINAL content that has NEVER been written before. This is not just another generic crypto thread - discover hidden angles, controversial takes, and unexplored perspectives within "{topic}" for {self.project_name}.
        
        ABSOLUTE REQUIREMENTS FOR UNIQUENESS:
        🚫 FORBIDDEN: Generic definitions, basic explanations, obvious benefits, standard comparisons, predictable structures
        ✅ REQUIRED: Contrarian insights, hidden mechanics, counterintuitive truths, fresh mental models, unexplored connections
        
        Choose ONE of these FRESH approaches (never pick the same twice):
        1. "The Hidden Cost Perspective" - What nobody talks about regarding {topic}
        2. "The Contrarian Take" - Challenge conventional wisdom about {topic}
        3. "The Technical Deep Dive" - Microscopic analysis of overlooked mechanics
        4. "The Historical Evolution" - How {topic} emerged and morphed in {self.project_name}
        5. "The Future Speculation" - Radical predictions about {topic}'s evolution
        6. "The Ecosystem Impact" - Ripple effects and unexpected consequences
        7. "The Developer's Secret" - Insider technical knowledge about {topic}
        8. "The Economic Game Theory" - Strategic implications and incentive analysis
        9. "The User Experience Lens" - How {topic} actually affects real users
        10. "The Philosophical Angle" - Deeper implications for crypto's future
        
        CREATIVE EXECUTION RULES:
        - Start with a SHOCKING statement or counterintuitive claim about {topic}
        - Use unexpected analogies (compare crypto concepts to biology, physics, psychology, etc.)
        - Reveal "secrets" or little-known technical details
        - Challenge popular assumptions with evidence from the research
        - Present familiar concepts through completely new frameworks
        - Ask provocative questions that make people think differently
        - End with mind-bending implications or predictions
        
        BANNED PHRASES/APPROACHES:
        - "Let's break down..."
        - "Here's why [topic] matters..."
        - "The benefits of..."
        - Any explanation that starts with basic definitions
        - Generic feature lists
        - Obvious use cases everyone knows
        
        Use this comprehensive research to find UNIQUE angles and hidden insights:
        {research_content}
        
        THREAD REQUIREMENTS:
        - Create exactly {thread_length} tweets that shock, educate, and provoke thought
        - Each tweet must reveal something most people don't know about {topic}
        - Build a narrative that's impossible to ignore
        - Include specific technical details presented in revolutionary ways
        - Make readers say "I never thought about it that way"
        
        Format as:
        1/{thread_length}: [mind-blowing opener with counterintuitive claim]
        2/{thread_length}: [supporting evidence that surprises]
        ...continue with escalating insights...
        {thread_length}/{thread_length}: [paradigm-shifting conclusion]
        
        Create content so original and insightful that it becomes the definitive thread about this aspect of {topic} in {self.project_name}.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
  
    def create_blog_post(self, topic: str, length: str = "medium") -> str:
        """Create blog post using comprehensive research"""
        if not self.vector_store:
            return "Error: No project setup. Run setup_project() first."
          
        print(f"Creating blog post about: {topic}")
      
        # Get even more comprehensive research for blog posts
        docs = self.vector_store.similarity_search(topic, k=35)
        research_content = "\n".join([doc.page_content for doc in docs])
      
        length_guide = {
            "short": "800-1200 words, 4-5 main sections",
            "medium": "1500-2500 words, 6-8 main sections", 
            "long": "2500-4000 words, 8-10 main sections"
        }
      
        target_length = length_guide.get(length, "1500-2500 words, 6-8 main sections")
        
        prompt = f"""
        REVOLUTIONARY CONTENT MANDATE: Create a GROUNDBREAKING blog post about "{topic}" in {self.project_name} that completely reframes how people think about this subject. This must be the most insightful piece ever written on this specific angle.
        
        ORIGINALITY IMPERATIVES:
        🚫 ABSOLUTELY FORBIDDEN: Standard introductions, obvious explanations, generic benefits/challenges, predictable structures, common examples, typical conclusions
        ✅ MANDATORY: Original thesis, contrarian perspectives, hidden connections, fresh frameworks, exclusive insights, paradigm shifts
        
        SELECT ONE REVOLUTIONARY FRAMEWORK (rotate between different posts):
        A) "The Iceberg Analysis" - Reveal the 90% hidden mechanics beneath surface understanding of {topic}
        B) "The Paradox Exploration" - Uncover contradictions and paradoxes within {topic} that reveal deeper truths
        C) "The Evolution Hypothesis" - Propose how {topic} is secretly evolving beyond current understanding
        D) "The System Dynamics" - Map the invisible forces and feedback loops driving {topic}
        E) "The Contrarian Thesis" - Build a compelling case against conventional wisdom about {topic}
        F) "The Convergence Theory" - Show how {topic} intersects with unexpected domains
        G) "The Future Archaeology" - Analyze {topic} as if looking back from 2030
        H) "The Mental Model Revolution" - Introduce completely new ways to think about {topic}
        I) "The Hidden Economics" - Expose the incentive structures and game theory within {topic}
        J) "The Technical Philosophy" - Explore the deeper implications of {topic}'s technical design
        
        CONTENT INNOVATION REQUIREMENTS:
        - Develop an original thesis that challenges existing thinking
        - Create new terminology or frameworks for understanding {topic}
        - Use research to support contrarian or counterintuitive claims
        - Present technical concepts through revolutionary analogies
        - Structure sections with provocative, non-obvious headings
        - Include exclusive insights not found anywhere else
        - Build to conclusions that shift paradigms
        
        Target: {target_length}
        
        Use this research to construct your revolutionary analysis:
        {research_content}
        
        STRUCTURE INNOVATION:
        - Hook: Start with a shocking revelation or counterintuitive claim about {topic}
        - Thesis: Present your original framework/theory about {topic}
        - Evidence: Use research to build your case with surprising insights
        - Implications: Explore consequences that others haven't considered
        - Paradigm Shift: Conclude with how this changes everything
        
        SECTION HEADING EXAMPLES (create similar original ones):
        - "The [Topic] Illusion: Why Everything You Think You Know Is Wrong"
        - "The Hidden Variables: [Topic]'s Secret Control Mechanisms"  
        - "Beyond [Topic]: The Invisible Forces Shaping the Future"
        - "The [Topic] Paradox: How Success Creates Its Own Problems"
        - "Deconstructing [Topic]: The Mental Models We Need to Abandon"
        
        BANNED GENERIC SECTIONS:
        - "What is [topic]?"
        - "Benefits of [topic]"
        - "How [topic] works"
        - "Use cases for [topic]"
        - "Future of [topic]"
        
        Create content so revolutionary that it becomes required reading for anyone serious about understanding {topic} in {self.project_name}. Make readers completely rethink their assumptions.
        """
      
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

def create_content_system(groq_api_key: str):
    """Initialize the content creation system"""
    return SimpleContentCreator(groq_api_key)