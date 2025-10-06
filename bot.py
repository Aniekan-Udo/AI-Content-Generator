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
    def __init__(self, API_KEY: str):
        if not API_KEY:
            raise ValueError("GROQ API key is required")
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
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
  
    def create_twitter_thread(self, topic: str) -> str:
        """Create Twitter thread using comprehensive research"""
        if not self.vector_store:
            return "Error: No project setup. Run setup_project() first."
        
        print(f"Creating Twitter thread about: {topic}")
    
        # Get comprehensive research from ALL documents
        docs = self.vector_store.similarity_search(topic, k=50)
        research_content = "\n".join([doc.page_content for doc in docs])
    
        prompt = f"""
        YOUR MISSION: Create a compelling, data-rich analysis of "{topic}" for {self.project_name} that reads like professional trading research - clear, bold, and impossible to ignore.
        
        ═══════════════════════════════════════════════════
        DEEP RESEARCH & DATA EXTRACTION
        ═══════════════════════════════════════════════════
        
        Thoroughly analyze all research materials:
        {research_content}
        
        Extract and note:
        - Specific numbers, percentages, rankings
        - Time periods and dates
        - Comparisons to competitors or alternatives
        - Volume metrics, user counts, transaction data
        - Any "first" or "only" claims you can make
        - Historical patterns or cycles
        - Concrete examples and case studies
        
        ═══════════════════════════════════════════════════
        CONTENT APPROACH - CHOOSE WHAT FITS BEST
        ═══════════════════════════════════════════════════
        
        Based on your research and what story the data tells, choose the most natural structure. Here are proven approaches you can use:
        
        **OPTION A - Classic Thesis-Driven** (great for big picture insights):
        - Open with *italicized thesis* that captures the main insight
        - Present **bold data points** that prove your thesis
        - Add bullet list of supporting evidence
        - Conclude with clear takeaway
        
        **OPTION B - Data-First Impact** (when you have shocking numbers):
        - Lead with **bold shocking statistic** 
        - Explain what this means and why it matters
        - Layer in additional context and evidence
        - End with implications
        
        **OPTION C - Narrative Flow** (for complex stories):
        - Set up the situation or context
        - Walk through the evidence chronologically or logically
        - Use formatting to highlight key moments
        - Build to a powerful conclusion
        
        **OPTION D - Comparison Framework** (for relative positioning):
        - Present the comparison or contrast
        - Break down the specific differences with data
        - Use bullets to list key differentiators
        - Synthesize what this reveals
        
        **OPTION E - Question-Answer** (for addressing misconceptions):
        - Pose the question or misconception
        - Systematically address it with evidence
        - Use formatting to emphasize key answers
        - Conclude with the truth
        
        You can also blend these approaches or create your own structure - whatever lets the data and insights flow most naturally for THIS specific topic.
        
        ═══════════════════════════════════════════════════
        FORMATTING TOOLS - USE STRATEGICALLY
        ═══════════════════════════════════════════════════
        
        *Italics* (use *asterisks*):
        - Core thesis statements
        - Key concepts or quotes
        - Emphasis on important phrases
        - Questions you're answering
        
        **Bold** (use **double asterisks**):
        - Your 1-2 most critical data points or claims
        - Numbers and metrics that prove your point
        - "Shocking" facts that make people stop scrolling
        - Strong opening statements when appropriate
        
        Bullet points (use - or •):
        - Lists of supporting evidence
        - Multiple factors or reasons
        - Easy-to-scan supporting points
        - When you have 4+ related items
        
        Paragraph breaks:
        - Create white space for readability
        - Separate different ideas
        - Guide the reader through your argument
        - Don't be afraid of single-sentence paragraphs for impact
        
        ═══════════════════════════════════════════════════
        WRITING PRINCIPLES (NON-NEGOTIABLE)
        ═══════════════════════════════════════════════════
        
        TONE: Confident, data-driven, assertive (not hype-y)
        - State facts with conviction
        - Let the data speak for itself
        - Avoid unnecessary qualifiers ("might," "could," "possibly")
        - Use present tense for immediacy
        
        LANGUAGE: Clear and accessible
        - Explain technical terms in parentheses immediately
        - Use analogies to familiar concepts when helpful
        - Short sentences for impact. Longer ones for context.
        - No jargon walls - a smart beginner should follow along
        
        SPECIFICITY: Always choose concrete over vague
        - "100x from $4m to $440m" NOT "grew significantly"
        - "3rd most-traded memecoin" NOT "very popular"
        - "4 months ago" NOT "recently"
        - "80% of the time" NOT "usually"
        
        ═══════════════════════════════════════════════════
        OUTPUT REQUIREMENTS
        ═══════════════════════════════════════════════════
        
        Create a SINGLE, LONG-FORM piece with:
        
        - 800-1500 characters total (substantial but readable)
        - Strategic formatting throughout (italics, bold, bullets where they fit)
        - Clear paragraph breaks for visual flow
        - A natural narrative that serves the content
        - At least 3-5 specific data points from research
        - Simple language that beginners can understand
        - Structure that feels organic to THIS topic
        
        ═══════════════════════════════════════════════════
        QUALITY CHECKLIST
        ═══════════════════════════════════════════════════
        
        Before submitting, verify:
        
        ✓ Structure feels natural and serves the content
        ✓ Formatting enhances readability (not forced or excessive)
        ✓ At least 5 specific data points or metrics included
        ✓ Clear paragraph breaks create visual breathing room
        ✓ Language is simple enough for crypto beginners
        ✓ Tone is confident and data-driven, not hype-y
        ✓ Every claim is backed by research materials
        ✓ Flow feels smooth and logical
        ✓ Would make someone stop scrolling and read carefully
        ✓ Doesn't feel formulaic or templated
        
        ═══════════════════════════════════════════════════
        
        Now create a compelling analysis of {topic} for {self.project_name}. Let the research guide your structure - choose whatever approach makes the data and insights shine most effectively. Make it impossible to ignore.
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