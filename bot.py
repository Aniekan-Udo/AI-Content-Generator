import os
import time
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CryptoContentCreator:
    def __init__(self, API_KEY: str):
        if not API_KEY:
            raise ValueError("GROQ API key is required")
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)
        
        # Use the robust embedding initialization
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True})
        
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
  
    def create_twitter_thread(self, topic: str, length: str = "medium") -> str:
        """Create educational crypto Twitter thread optimized for Kaito rewards
        
        Args:
            topic: The topic to write about
            length: Tweet length - 'short' (400-600 chars), 'medium' (600-850 chars), or 'long' (900-1500 chars)
        """
        if not self.vector_store:
            return "Error: No project setup. Run setup_project() first."
          
        print(f"Creating {length} educational crypto thread about: {topic}")
      
        # Get comprehensive research from ALL documents
        docs = self.vector_store.similarity_search(topic, k=50)
        research_content = "\n".join([doc.page_content for doc in docs])
        
        # Define length specifications
        length_specs = {
            "short": {
                "chars": "400-600 characters (concise and punchy)",
                "data_points": "3-4 specific data points",
                "style": "Quick, impactful insights. Get to the point fast."
            },
            "medium": {
                "chars": "600-850 characters (moderate, balanced)",
                "data_points": "4-6 specific data points",
                "style": "Balanced depth with clarity. Comfortable reading length."
            },
            "long": {
                "chars": "900-1500 characters (comprehensive and detailed)",
                "data_points": "6-8 specific data points",
                "style": "Deep dive with full context. Maximum educational value."
            }
        }
        
        spec = length_specs.get(length, length_specs["medium"])
      
        prompt = f"""
        YOUR MISSION: Create an educational crypto Twitter thread about "{topic}" for {self.project_name} that teaches readers something valuable they didn't know before. Optimize for Kaito's educational rewards by combining genuine learning value with engaging delivery.
        
        ═══════════════════════════════════════════════════
        STEP 1: DEEP RESEARCH & DATA EXTRACTION
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
        - Technical mechanics or how things work
        
        ═══════════════════════════════════════════════════
        STEP 2: IDENTIFY THE KNOWLEDGE GAP
        ═══════════════════════════════════════════════════
        
        Find what people DON'T understand about {topic}:
        - What misconception is most common?
        - What hidden mechanism does nobody talk about?
        - What principle applies here that people miss?
        - What would change someone's strategy if they knew it?
        - What counter-intuitive truth does data reveal?
        
        This gap is your teaching opportunity.
        
        ═══════════════════════════════════════════════════
        STEP 3: BUILD YOUR EDUCATIONAL NARRATIVE
        ═══════════════════════════════════════════════════
        
        Structure your content like this:
        
        **OPENING:**
        - Start with something that catches attention: a misconception, gap, or interesting observation
        - Could be: "Most people think...", "Here's what nobody talks about...", "Ever wonder why...", or just "Something's off about {topic}"
        - Or jump straight into a surprising fact if it's compelling
        - Make it conversational—this can be relaxed and natural
        
        **BODY (build naturally):**
        - Teach the insight or principle, but don't force a rigid structure
        - Use data and examples when they strengthen understanding
        - Highlight key points in **bold** or *italics* where it feels right
        - Connect concepts to real market behavior or events
        - You can use bullets if it makes sense, but they're optional
        - Let the narrative flow—don't check boxes
        
        **CLOSING:**
        - Leave them with the core takeaway or principle
        - Could be short, could be a few sentences
        - Practical insight: "This is why...", "Watch for...", "Now you see why..."
        - Or just end with the lesson itself if it stands alone
        
        ═══════════════════════════════════════════════════
        KAITO EDUCATIONAL OPTIMIZATION
        ═══════════════════════════════════════════════════
        
        Kaito rewards high-quality educational content. Maximize this by:
        
        SEMANTIC VALUE:
        - Every statement teaches something (no fluff)
        - Use precise terminology correctly (shows expertise)
        - Connect concepts to broader crypto principles
        - Explain the "why" behind mechanisms, not just the "what"
        
        CREDIBILITY SIGNALS:
        - Back claims with specific data from research
        - Reference actual events, projects, or timeframes
        - Show depth of understanding
        - Acknowledge complexity where it exists
        
        Structure as a lesson with clear progression, but keep it natural:
        - Identify and correct misconceptions when relevant
        - Teach principles that apply broadly, not just surface facts
        - Make the learning stick, but don't force a rigid format
        - Let the best way to communicate emerge—sometimes that's bullets, sometimes it's narrative
        
        ═══════════════════════════════════════════════════
        TONE & LANGUAGE
        ═══════════════════════════════════════════════════
        
        TONE: Expert sharing insight naturally, not lecturing
        - Conversational but credible
        - Friendly and approachable
        - Sound like you know what you're talking about without being stiff
        - It's okay to be casual—you're teaching, not giving a speech
        
        CRYPTO-SPECIFIC LANGUAGE:
        - Use crypto terms naturally and correctly (shows credibility)
        - Explain technical concepts in plain language
        - Reference market dynamics, cycles, incentives
        - Connect to how people actually trade/invest
        
        CLARITY OVER CLEVERNESS:
        - Prioritize understanding over entertainment
        - Short sentences for complex ideas
        - Paragraph breaks for readability
        - Use analogies that clarify, not confuse
        
        ═══════════════════════════════════════════════════
        FORMATTING FOR EDUCATIONAL IMPACT
        ═══════════════════════════════════════════════════
        
        *Italics* (use *asterisks*):
        - Core principles or the main lesson
        - The key insight being taught
        - What readers should remember
        
        **Bold** (use **double asterisks**):
        - Critical data points that prove your lesson
        - Numbers that support your teaching
        - The "aha" facts
        
        Bullet points (use - or •):
        - Distinct concepts or principles
        - Key takeaways to remember
        - Easy reference for the lesson
        
        Paragraph breaks:
        - Separate different concepts
        - Create breathing room for learning
        - Guide readers through the lesson progression
        
        ═══════════════════════════════════════════════════
        OUTPUT REQUIREMENTS - {length.upper()} LENGTH
        ═══════════════════════════════════════════════════
        
        Create a SINGLE piece with these specifications:
        
        LENGTH: {spec['chars']}
        DATA POINTS: At least {spec['data_points']} from research
        STYLE: {spec['style']}
        
        - Include ALL core information and key insights
        - Educational focus: teaches something concrete
        - Natural, engaging flow (not formulaic)
        - Complete thoughts and explanations (don't cut corners)
        - Pack maximum value into the specified space
        
        IMPORTANT: Ensure the tweet is SELF-CONTAINED with all essential information. Don't leave readers needing more context. All core insights, data points, and the main lesson should be present in this single tweet.
        
        ═══════════════════════════════════════════════════
        QUALITY CHECKLIST FOR KAITO
        ═══════════════════════════════════════════════════
        
        Before submitting, verify:
        
        ✓ Opens with a clear learning opportunity or misconception to correct
        ✓ Teaches a principle, mechanism, or pattern (not just facts)
        ✓ Includes the specified number of data points from research
        ✓ Explains the "why" behind concepts, not just the "what"
        ✓ Provides real crypto examples or applications
        ✓ Uses accurate terminology (shows credibility for Kaito)
        ✓ Identifies a knowledge gap most traders/investors have
        ✓ Concludes with actionable understanding
        ✓ Formatted for easy scanning and reference
        ✓ Every claim backed by research materials
        ✓ Contains ALL core information (self-contained)
        ✓ Matches the {length.upper()} length specification ({spec['chars']})
        ✓ Would make someone want to screenshot and save it
        ✓ Teaches something valuable, not just entertaining
        ✓ Could serve as educational reference material
        ✓ Demonstrates expertise and depth
        
        ═══════════════════════════════════════════════════
        
        Now create an educational crypto thread about {topic} for {self.project_name} that teaches readers something valuable they didn't know. Make it {length.upper()} LENGTH ({spec['chars']}), clear, credible, and worth saving. Include ALL core information in this single tweet. Optimize for Kaito's educational rewards by combining genuine learning value with engaging delivery that makes complex concepts understandable.
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
    return CryptoContentCreator(groq_api_key)