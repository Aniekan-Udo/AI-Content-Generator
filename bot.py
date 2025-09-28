# bot.py

import os
import time
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# IMPORTANT: Import Document here to correctly type hint the input
from langchain_core.documents import Document 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# IMPORTANT: Replace with environment variable loading in a real application
groq_api_key = os.getenv("GROQ_API_KEY")


class SimpleContentCreator:
    def __init__(self, groq_api_key: str):
         self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'})
         self.text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=1000,
             chunk_overlap=200
         )

         self.project_name = None
         self.vector_store = None
        
    # --- MODIFIED FUNCTION: Accepts pre-split documents ---
    def setup_project(self, project_name: str, all_documents: List[Document]):
        """
        Setup project by creating a vector store from pre-loaded and split documents. 
        Loading/splitting logic is now handled by app.py.
        """
        print(f"Setting up project: {project_name}")
        
        self.project_name = project_name
        
        # Create vector store with all content
        if all_documents:
            print(f"Creating knowledge base with {len(all_documents)} document chunks")
            
            # --- FIX FOR VALIDATION ERROR ---
            # Clean the project name: replace spaces/underscores with hyphens 
            # for safe collection naming (Chroma)
            clean_project_name = project_name.lower().replace(' ', '-').replace('_', '-')
            
            self.vector_store = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                # Use the cleaned name for the collection
                collection_name=f"{clean_project_name}-{int(time.time())}" 
            )
            # --- END FIX FOR VALIDATION ERROR ---
            
            print(f"Setup complete for {project_name}")
        else:
            print("Warning: No documents were loaded. Vector store is not created.")
            self.vector_store = None # Explicitly set to None if empty

    
    def create_twitter_thread(self, topic: str, thread_length: int = 6) -> str:
        """Create Twitter thread using comprehensive research"""
        if not self.vector_store:
            return "Error: No project setup. Run setup_project() first."
            
        print(f"Creating Twitter thread about: {topic}")
        
        # Get comprehensive research from ALL documents
        docs = self.vector_store.similarity_search(topic, k=25)
        research_content = "\n".join([doc.page_content for doc in docs])
        
        # NEW PROMPT (Truncated for brevity, assuming existing logic is fine)
        prompt = f"""
CRITICAL CREATIVITY MANDATE: You MUST create completely ORIGINAL content that has NEVER been written before. This is not just another generic crypto thread - discover hidden angles, controversial takes, and unexplored perspectives within "{topic}" for {self.project_name}.

ABSOLUTE REQUIREMENTS FOR UNIQUENESS:
🚫 FORBIDDEN: Generic definitions, basic explanations, obvious benefits, standard comparisons, predictable structures
✅ REQUIRED: Contrarian insights, hidden mechanics, counterintuitive truths, fresh mental models, unexplored connections

... [Rest of the prompt remains the same] ...

Use this comprehensive research to find UNIQUE angles and hidden insights:
{research_content}

THREAD REQUIREMENTS:
- Create exactly {thread_length} tweets that shock, educate, and provoke thought
- Each tweet must reveal something most people don't know about {topic}

Format as:
1/{thread_length}: [mind-blowing opener with counterintuitive claim]
2/{thread_length}: [supporting evidence that surprises]
...
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
        
        # NEW PROMPT (Truncated for brevity, assuming existing logic is fine)
        prompt = f"""
REVOLUTIONARY CONTENT MANDATE: Create a GROUNDBREAKING blog post about "{topic}" in {self.project_name} that completely reframes how people think about this subject. This must be the most insightful piece ever written on this specific angle.

ORIGINALITY IMPERATIVES:
🚫 ABSOLUTELY FORBIDDEN: Standard introductions, obvious explanations, generic benefits/challenges, predictable structures, common examples, typical conclusions
✅ MANDATORY: Original thesis, contrarian perspectives, hidden connections, fresh frameworks, exclusive insights, paradigm shifts

... [Rest of the prompt remains the same] ...

Target: {target_length}

Use this research to construct your revolutionary analysis:
{research_content}

STRUCTURE INNOVATION:
- Hook: Start with a shocking revelation or counterintuitive claim about {topic}
- Thesis: Present your original framework/theory about {topic}
- Evidence: Use research to build your case with surprising insights
- Implications: Explore consequences that others haven't considered
- Paradigm Shift: Conclude with how this changes everything

BANNED GENERIC SECTIONS:
- "What is [topic]?"
- "Benefits of [topic]"

Create content so revolutionary that it becomes required reading for anyone serious about understanding {topic} in {self.project_name}.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

# Simple usage functions
def create_content_system(groq_api_key: str):
    """Initialize the content creation system"""
    return SimpleContentCreator(groq_api_key)

# import os
# import time
# from typing import List
# from langchain_community.document_loaders import WebBaseLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.messages import HumanMessage
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# groq_api_key="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"

# class SimpleContentCreator:
#     def __init__(self, groq_api_key: str):
#         self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         self.project_name = None
#         self.vector_store = None
        
#     def setup_project(self, project_name: str, urls: List[str], whitepaper_path: str = None):
#         """Setup project by loading documents from specified URLs and whitepaper"""
#         print(f"Setting up project: {project_name}")
#         print(f"Processing {len(urls)} URLs...")
        
#         self.project_name = project_name
#         all_documents = []
        
#         # Process each URL directly (no automatic subpage discovery)
#         for url in urls:
#             print(f"Processing: {url}")
#             try:
#                 loader = WebBaseLoader([url])
#                 docs = loader.load()
#                 chunks = self.text_splitter.split_documents(docs)
#                 all_documents.extend(chunks)
#                 print(f"  Loaded {len(chunks)} chunks from {url}")
#             except Exception as e:
#                 print(f"  Error loading {url}: {str(e)}")
            
#         # Process whitepaper if provided
#         if whitepaper_path and os.path.exists(whitepaper_path):
#             print(f"Processing whitepaper: {whitepaper_path}")
#             try:
#                 loader = TextLoader(whitepaper_path)
#                 docs = loader.load()
#                 chunks = self.text_splitter.split_documents(docs)
#                 all_documents.extend(chunks)
#                 print(f"  Loaded {len(chunks)} chunks from whitepaper")
#             except Exception as e:
#                 print(f"  Error loading whitepaper: {str(e)}")
            
#         # Create vector store with all content
#         if all_documents:
#             print(f"Creating knowledge base with {len(all_documents)} document chunks")
#             self.vector_store = Chroma.from_documents(
#                 documents=all_documents,
#                 embedding=self.embeddings,
#                 collection_name=f"{project_name.lower()}_{int(time.time())}"
#             )
#             print(f"Setup complete for {project_name}")
#         else:
#             print("Warning: No documents were loaded")

    
#     def create_twitter_thread(self, topic: str, thread_length: int = 6) -> str:
#         """Create Twitter thread using comprehensive research"""
#         if not self.vector_store:
#             return "Error: No project setup. Run setup_project() first."
            
#         print(f"Creating Twitter thread about: {topic}")
        
#         # Get comprehensive research from ALL documents
#         docs = self.vector_store.similarity_search(topic, k=25)
#         research_content = "\n".join([doc.page_content for doc in docs])
        
#         #ORIGINAL PROMPT
# #         prompt = f"""
# # Create a high-quality crypto Twitter thread about "{topic}" for {self.project_name} following Kaito guidelines.

# # IMPORTANT: Don't just focus on "{topic}" literally. Use ALL the research below to find related concepts, underlying mechanisms, technical details, and broader context that would make the thread more comprehensive and educational.
# # Don't repeat or recycle tweets, be creative in crafting your tweets to avoid repetition

# # For example: If asked about "{self.project_name}", also include consensus mechanisms, protocol architecture, validator systems, tokenomics, security features, or any other relevant technical aspects found in the research.

# # Use ALL this comprehensive research from the project's documentation:
# # {research_content}

# # KAITO GUIDELINES COMPLIANCE:
# # - Focus on crypto-related content backed by deep knowledge from the research
# # - Create high-quality, long-form content about this specific crypto protocol ({self.project_name})
# # - Provide original insightful analysis based on the comprehensive research provided
# # - Ensure content is relevant and timely for crypto community
# # - Make it educational and valuable, not random commentary

# # THREAD REQUIREMENTS:
# # - Create exactly {thread_length} tweets that form a high-quality, long-form educational thread
# # - Each tweet should be SUBSTANTIAL (close to 280 characters) with detailed technical content
# # - Use insights from ALL sources and subtopics in the research
# # - Always simplify tweets, so audience can understand what you're talking about
# # - Include specific technical details, examples, and deep analysis in each tweet
# # - Explore related concepts found in the research even if not explicitly mentioned in "{topic}"
# # - Build knowledge progressively through the thread with comprehensive explanations
# # - Make each tweet a mini-lesson with actionable insights
# # - Focus on protocol-specific insights that demonstrate deep technical understanding
# # - Avoid brief summaries - provide detailed explanations and analysis

# # CONTENT STRATEGY:
# # - Start with a compelling hook about {topic} in {self.project_name}
# # - Explore underlying technical mechanisms found in the research
# # - Include related concepts like consensus, architecture, security, tokenomics as relevant
# # - Present technical analysis backed by the research
# # - Include specific examples and use cases from documentation
# # - Explain complex concepts clearly
# # - End with key insights or broader implications

# # Format as:
# # 1/{thread_length}: [tweet content]
# # 2/{thread_length}: [tweet content]
# # ...and so on

# # Create an insightful, knowledge-backed thread that goes beyond just "{topic}" to explore the comprehensive technical ecosystem found in the research.
# # """
#         #NEW PROMPT
#         prompt = f"""
# CRITICAL CREATIVITY MANDATE: You MUST create completely ORIGINAL content that has NEVER been written before. This is not just another generic crypto thread - discover hidden angles, controversial takes, and unexplored perspectives within "{topic}" for {self.project_name}.

# ABSOLUTE REQUIREMENTS FOR UNIQUENESS:
# 🚫 FORBIDDEN: Generic definitions, basic explanations, obvious benefits, standard comparisons, predictable structures
# ✅ REQUIRED: Contrarian insights, hidden mechanics, counterintuitive truths, fresh mental models, unexplored connections

# Choose ONE of these FRESH approaches (never pick the same twice):
# 1. "The Hidden Cost Perspective" - What nobody talks about regarding {topic}
# 2. "The Contrarian Take" - Challenge conventional wisdom about {topic}
# 3. "The Technical Deep Dive" - Microscopic analysis of overlooked mechanics
# 4. "The Historical Evolution" - How {topic} emerged and morphed in {self.project_name}
# 5. "The Future Speculation" - Radical predictions about {topic}'s evolution
# 6. "The Ecosystem Impact" - Ripple effects and unexpected consequences
# 7. "The Developer's Secret" - Insider technical knowledge about {topic}
# 8. "The Economic Game Theory" - Strategic implications and incentive analysis
# 9. "The User Experience Lens" - How {topic} actually affects real users
# 10. "The Philosophical Angle" - Deeper implications for crypto's future

# CREATIVE EXECUTION RULES:
# - Start with a SHOCKING statement or counterintuitive claim about {topic}
# - Use unexpected analogies (compare crypto concepts to biology, physics, psychology, etc.)
# - Reveal "secrets" or little-known technical details
# - Challenge popular assumptions with evidence from the research
# - Present familiar concepts through completely new frameworks
# - Ask provocative questions that make people think differently
# - End with mind-bending implications or predictions

# BANNED PHRASES/APPROACHES:
# - "Let's break down..."
# - "Here's why [topic] matters..."
# - "The benefits of..."
# - Any explanation that starts with basic definitions
# - Generic feature lists
# - Obvious use cases everyone knows

# Use this comprehensive research to find UNIQUE angles and hidden insights:
# {research_content}

# THREAD REQUIREMENTS:
# - Create exactly {thread_length} tweets that shock, educate, and provoke thought
# - Each tweet must reveal something most people don't know about {topic}
# - Build a narrative that's impossible to ignore
# - Include specific technical details presented in revolutionary ways
# - Make readers say "I never thought about it that way"

# Format as:
# 1/{thread_length}: [mind-blowing opener with counterintuitive claim]
# 2/{thread_length}: [supporting evidence that surprises]
# ...continue with escalating insights...
# {thread_length}/{thread_length}: [paradigm-shifting conclusion]

# Create content so original and insightful that it becomes the definitive thread about this aspect of {topic} in {self.project_name}.
# """
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         return response.content
    
#     def create_blog_post(self, topic: str, length: str = "medium") -> str:
#         """Create blog post using comprehensive research"""
#         if not self.vector_store:
#             return "Error: No project setup. Run setup_project() first."
            
#         print(f"Creating blog post about: {topic}")
        
#         # Get even more comprehensive research for blog posts
#         docs = self.vector_store.similarity_search(topic, k=35)
#         research_content = "\n".join([doc.page_content for doc in docs])
        
#         length_guide = {
#             "short": "800-1200 words, 4-5 main sections",
#             "medium": "1500-2500 words, 6-8 main sections", 
#             "long": "2500-4000 words, 8-10 main sections"
#         }
        
#         target_length = length_guide.get(length, "1500-2500 words, 6-8 main sections")
#         #ORIGINAL PROMPT
# #         prompt = f"""
# # Write a comprehensive blog post about "{topic}" for {self.project_name}.

# # IMPORTANT: Don't just focus on "{topic}" literally. Use ALL the research below to find related concepts, underlying mechanisms, technical details, and broader context that would make the blog post more comprehensive and educational.
# # Don't repeat or recycle contents, be creative in crafting your contents to avoid repetition
# # For example: If asked about "{self.project_name}", also explore consensus mechanisms, protocol architecture, validator systems, tokenomics, security features, governance, scalability solutions, or any other relevant technical aspects found in the research.

# # Use ALL this comprehensive research from the project's documentation:
# # {research_content}

# # BLOG POST REQUIREMENTS:
# # - Target: {target_length}
# # - Use insights from ALL sources and subtopics in the research
# # - Explore related concepts found in the research even if not explicitly mentioned in "{topic}"
# # - Include specific examples and use cases from the research
# # - Create proper blog structure with headings
# # - Add image placeholders where helpful: [IMAGE: specific description]
# # - Educational and authoritative tone
# # - Technical depth appropriate for crypto/blockchain audience

# # CONTENT STRATEGY:
# # - Start with compelling introduction about {topic} and {self.project_name}
# # - Explore underlying technical mechanisms and related concepts found in the research
# # - Include sections on architecture, consensus, security, tokenomics, governance as relevant
# # - Combine different aspects, perspectives, and technical details from ALL research
# # - Present comprehensive analysis that goes beyond surface-level {topic} discussion
# # - Include practical applications and real-world implications
# # - Technical explanations with specific examples from the documentation

# # STRUCTURE APPROACH:
# # 1. Compelling introduction that hooks readers about {topic} and broader context
# # 2. Multiple detailed sections covering different subtopics discovered in research
# # 3. Technical explanations with specific examples from ALL available research
# # 4. Practical applications and use cases across related areas
# # 5. Comprehensive analysis of implications and future considerations
# # 6. Conclusion with key takeaways from the holistic exploration

# # Analyze all the research content, identify key subtopics and themes beyond just "{topic}", and create a comprehensive blog post that thoroughly explores the broader technical ecosystem using ALL available documentation.
# # """
#         #NEW PROMPT
#         prompt = f"""
# REVOLUTIONARY CONTENT MANDATE: Create a GROUNDBREAKING blog post about "{topic}" in {self.project_name} that completely reframes how people think about this subject. This must be the most insightful piece ever written on this specific angle.

# ORIGINALITY IMPERATIVES:
# 🚫 ABSOLUTELY FORBIDDEN: Standard introductions, obvious explanations, generic benefits/challenges, predictable structures, common examples, typical conclusions
# ✅ MANDATORY: Original thesis, contrarian perspectives, hidden connections, fresh frameworks, exclusive insights, paradigm shifts

# SELECT ONE REVOLUTIONARY FRAMEWORK (rotate between different posts):

# A) "The Iceberg Analysis" - Reveal the 90% hidden mechanics beneath surface understanding of {topic}
# B) "The Paradox Exploration" - Uncover contradictions and paradoxes within {topic} that reveal deeper truths
# C) "The Evolution Hypothesis" - Propose how {topic} is secretly evolving beyond current understanding
# D) "The System Dynamics" - Map the invisible forces and feedback loops driving {topic}
# E) "The Contrarian Thesis" - Build a compelling case against conventional wisdom about {topic}
# F) "The Convergence Theory" - Show how {topic} intersects with unexpected domains
# G) "The Future Archaeology" - Analyze {topic} as if looking back from 2030
# H) "The Mental Model Revolution" - Introduce completely new ways to think about {topic}
# I) "The Hidden Economics" - Expose the incentive structures and game theory within {topic}
# J) "The Technical Philosophy" - Explore the deeper implications of {topic}'s technical design

# CONTENT INNOVATION REQUIREMENTS:
# - Develop an original thesis that challenges existing thinking
# - Create new terminology or frameworks for understanding {topic}
# - Use research to support contrarian or counterintuitive claims
# - Present technical concepts through revolutionary analogies
# - Structure sections with provocative, non-obvious headings
# - Include exclusive insights not found anywhere else
# - Build to conclusions that shift paradigms

# Target: {target_length}

# Use this research to construct your revolutionary analysis:
# {research_content}

# STRUCTURE INNOVATION:
# - Hook: Start with a shocking revelation or counterintuitive claim about {topic}
# - Thesis: Present your original framework/theory about {topic}
# - Evidence: Use research to build your case with surprising insights
# - Implications: Explore consequences that others haven't considered
# - Paradigm Shift: Conclude with how this changes everything

# SECTION HEADING EXAMPLES (create similar original ones):
# - "The [Topic] Illusion: Why Everything You Think You Know Is Wrong"
# - "The Hidden Variables: [Topic]'s Secret Control Mechanisms"  
# - "Beyond [Topic]: The Invisible Forces Shaping the Future"
# - "The [Topic] Paradox: How Success Creates Its Own Problems"
# - "Deconstructing [Topic]: The Mental Models We Need to Abandon"

# BANNED GENERIC SECTIONS:
# - "What is [topic]?"
# - "Benefits of [topic]"
# - "How [topic] works"
# - "Use cases for [topic]"
# - "Future of [topic]"

# Create content so revolutionary that it becomes required reading for anyone serious about understanding {topic} in {self.project_name}. Make readers completely rethink their assumptions.
# """

        
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         return response.content

# # Simple usage functions
# def create_content_system(groq_api_key: str):
#     """Initialize the content creation system"""
#     return SimpleContentCreator(groq_api_key)
