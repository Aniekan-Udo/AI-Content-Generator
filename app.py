import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules with fallback handling
import operator
from typing import Annotated
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Handle embeddings import with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_NEW = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_NEW = False

from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel as PydanticBaseModel, Field

from typing import TypedDict, Annotated
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import operator
from langchain.docstore.document import Document
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import WebBaseLoader

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tweepy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Thread Creator API",
    description="Create engaging Twitter threads using AI and Cysic documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class TweetRequest(BaseModel):
    query: str = Field(..., description="The topic or query for tweet generation")
    tweet_count: Optional[int] = Field(default=1, ge=1, le=10, description="Number of tweets to generate")
    style: Optional[str] = Field(default="engaging", description="Style of tweets: engaging, educational, technical")

class TweetResponse(BaseModel):
    success: bool
    tweets: List[str]
    metadata: Dict[str, Any]
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

# Global variables for the workflow
workflow_app = None
embeddings = None
txt_retriever = None
url_retriever = None
tweet_retriever = None

class DummyRetriever:
    """Dummy retriever for when real retrievers fail"""
    def get_relevant_documents(self, query):
        return [Document(page_content="Sample content for testing", metadata={"source": "dummy"})]

def filter_meaningful_tweets(tweets):
    """Filter tweets for meaningful content"""
    if not tweets:
        return []
    
    meaningful_tweets = []
    
    for tweet in tweets:
        if not isinstance(tweet, str) or len(tweet.strip()) < 10:
            continue
            
        # Skip spam-like content
        mention_count = tweet.count('@')
        hashtag_count = tweet.count('#')
        if mention_count > 5 or hashtag_count > 8:
            continue
            
        spam_keywords = ['buy now', 'limited time', 'click here', 'dm me']
        if any(keyword in tweet.lower() for keyword in spam_keywords):
            continue
            
        meaningful_keywords = ['learn', 'technology', 'cysic', 'blockchain', 'crypto']
        if (any(keyword in tweet.lower() for keyword in meaningful_keywords) or 
            len(tweet) > 80):
            meaningful_tweets.append(tweet.strip())
    
    return meaningful_tweets

def initialize_components():
    """Initialize all the components needed for the workflow"""
    global workflow_app, embeddings, txt_retriever, url_retriever, tweet_retriever
    
    try:
        logger.info("Initializing components...")
        
        # Get API key from environment or use the provided one
        api_key = os.getenv("GROQ_API_KEY", "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")
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        
        # Initialize embeddings with fallback
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            logger.info(f"Embeddings initialized successfully (New: {HUGGINGFACE_NEW})")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise e
        
        # Initialize retrievers
        setup_retrievers()
        
        # Initialize tools
        setup_tools(llm)
        
        # Create workflow
        workflow_app = create_workflow(llm)
        
        logger.info("Components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

def setup_retrievers():
    """Setup document retrievers"""
    global txt_retriever, url_retriever, tweet_retriever
    
    # Text splitter
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    # --- Local TXT Retriever ---
    try:
        # Check if the whitepaper file exists
        whitepaper_path = Path("Cysic whitepaper.txt")
        if whitepaper_path.exists():
            doc_loader = TextLoader(str(whitepaper_path))
            docs = doc_loader.load()
            docs_chunk = text_splitter.split_documents(docs)
            
            txt_store = Chroma.from_documents(
                documents=docs_chunk,
                collection_name="text_docs",
                embedding=embeddings,
                persist_directory="./chroma_store_txt"
            )
            txt_retriever = txt_store.as_retriever(search_kwargs={"k": 5})
        else:
            logger.warning("Whitepaper file not found, creating dummy retriever")
            txt_retriever = DummyRetriever()
            
    except Exception as e:
        logger.error(f"Failed to setup text retriever: {e}")
        txt_retriever = DummyRetriever()
    
    # --- URL Retriever ---
    try:
        base_urls = [
            "https://docs.cysic.xyz/readme/cysic-agent-to-agent-protocol",
            "https://docs.cysic.xyz/"
        ]
        url_loader = WebBaseLoader(base_urls)
        url_docs = url_loader.load()
        url_chunks = text_splitter.split_documents(url_docs)
        
        url_store = Chroma.from_documents(
            documents=url_chunks,
            collection_name="url_docs",
            embedding=embeddings,
            persist_directory="./chroma_store_url"
        )
        url_retriever = url_store.as_retriever(search_kwargs={"k": 5})
        
    except Exception as e:
        logger.error(f"Failed to setup URL retriever: {e}")
        url_retriever = DummyRetriever()
    
    # --- Tweet Retriever with sample data ---
    try:
        setup_sample_tweet_retriever(text_splitter)
    except Exception as e:
        logger.error(f"Failed to setup tweet retriever: {e}")
        tweet_retriever = DummyRetriever()

def setup_sample_tweet_retriever(text_splitter):
    """Setup tweet retriever with sample data"""
    global tweet_retriever
    
    sample_tweets = [
        "Cysic's Agent-to-Agent Protocol eliminates traditional blockchain bottlenecks by implementing direct peer verification through advanced zero-knowledge proofs, achieving sub-second finality.",
        "The protocol's recursive SNARK verification batches thousands of transactions per proof, enabling 100,000+ TPS while maintaining full decentralization and security.",
        "Understanding Cysic's technical architecture: how zkSNARKs enable instant settlement without compromising blockchain security or decentralization principles.",
        "Real-world impact: Cysic's technology reduces transaction costs by 95% compared to Ethereum L1 while enabling micropayments for IoT devices and high-frequency DeFi operations.",
        "Building the future of Web3 infrastructure with Cysic's innovative consensus mechanism that solves the blockchain trilemma through mathematical proofs rather than trade-offs."
    ]
    
    tweet_docs = [Document(page_content=t, metadata={"source": "sample"}) 
                 for t in sample_tweets]
    
    tweet_store = Chroma.from_documents(
        embedding=embeddings,
        collection_name="sample_tweets",
        documents=tweet_docs,
        persist_directory="./chroma_store_sample"
    )
    tweet_retriever = tweet_store.as_retriever(search_kwargs={"k": 3})

def setup_tools(llm):
    """Setup all the tools used in the workflow"""
    global text_loader, url_loader, tweet_loader, content_synthesizer, tweet_creator, quality_reviewer, kaito_compliance_checker, final_formatter
    
    MAX_OUTPUT_CHARS = 4000
    
    @tool
    def text_loader(query: str) -> str:
        """Retrieve relevant content from local TXT (Cysic whitepaper)."""
        try:
            results = txt_retriever.get_relevant_documents(query)
            if not results:
                return "No relevant content found in whitepaper."
            
            content = "\n---\n".join([doc.page_content for doc in results])
            return f"WHITEPAPER CONTENT:\n{content[:MAX_OUTPUT_CHARS]}"
        except Exception as e:
            return f"Error retrieving whitepaper content: {str(e)}"

    @tool
    def url_loader(query: str) -> str:
        """Retrieve relevant content from Cysic documentation URLs."""
        try:
            results = url_retriever.get_relevant_documents(query)
            if not results:
                return "No relevant content found in documentation."
            
            content = "\n---\n".join([doc.page_content for doc in results])
            return f"DOCUMENTATION CONTENT:\n{content[:MAX_OUTPUT_CHARS]}"
        except Exception as e:
            return f"Error retrieving documentation: {str(e)}"

    @tool
    def tweet_loader(query: str) -> str:
        """Retrieve meaningful tweet examples."""
        try:
            results = tweet_retriever.get_relevant_documents(query)
            if not results:
                return "No relevant tweets found."
            
            content = "\n---\n".join([doc.page_content for doc in results])
            return f"TWEET EXAMPLES:\n{content[:MAX_OUTPUT_CHARS]}"
        except Exception as e:
            return f"Error retrieving tweets: {str(e)}"

    @tool
    def content_synthesizer(retrieved_content: str) -> str:
        """Synthesize content into detailed tweet themes with specific information."""
        prompt = f"""
        You are analyzing rich content from Cysic's documentation, whitepaper, and community discussions. 
        Extract the most compelling, specific, and detailed information to create substantial tweet content.

        RETRIEVED CONTENT:
        {retrieved_content}

        Create 3-5 detailed tweet concepts that include:

        1. **SPECIFIC TECHNICAL DETAILS**: Extract exact technical innovations, numbers, protocols, features
        2. **CONCRETE EXAMPLES**: Real use cases, implementations, or applications mentioned
        3. **UNIQUE VALUE PROPOSITIONS**: What makes Cysic different from competitors
        4. **FACTUAL CLAIMS**: Specific benefits, performance metrics, or capabilities
        5. **EDUCATIONAL INSIGHTS**: Deep explanations that teach something valuable

        For each concept, provide:
        - The core factual insight (with specific details from the content)
        - Supporting technical explanation
        - Real-world application or benefit
        - Why this matters to the crypto/blockchain community

        DO NOT create generic themes. USE THE ACTUAL INFORMATION from the retrieved content.
        Focus on substance over style - we want information-dense, valuable content.
        """
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error in content synthesis: {str(e)}"

    @tool
    def tweet_creator(content: str) -> str:
        """Create information-rich, substantial tweets from synthesized content."""
        prompt = f"""
        Transform this detailed content into compelling, information-packed tweets.

        SYNTHESIZED CONTENT:
        {content}

        Create 2-3 substantial tweets that are INFORMATION-DENSE and VALUE-PACKED:

        REQUIREMENTS:
        ✅ LEAD WITH SPECIFIC FACTS: Start with concrete technical details, not questions
        ✅ INCLUDE EXACT INFORMATION: Use specific features, numbers, capabilities from the content
        ✅ EXPLAIN HOW/WHY: Don't just state what, explain the mechanism or reasoning
        ✅ PROVIDE CONTEXT: Why this matters, what problem it solves
        ✅ USE THREAD FORMAT: Each tweet should build on the previous one
        ✅ MAXIMIZE CHARACTER COUNT: Use close to 280 characters with valuable information
        ✅ INCLUDE TECHNICAL DEPTH: Show genuine expertise and understanding

        STRUCTURE EACH TWEET:
        - Hook: Bold factual statement or counterintuitive insight
        - Body: Detailed explanation with specifics
        - Context: Why this matters or what it enables
        - Optional: Implication or next step

        AVOID:
        ❌ Starting with questions
        ❌ Generic statements without specifics
        ❌ Vague claims without backing details
        ❌ Filler words that waste character count
        ❌ Asking "What do you think?" type endings

        EXAMPLE GOOD START:
        "Cysic's Agent-to-Agent Protocol eliminates the 40% overhead traditional blockchains face with cross-chain transactions by implementing direct peer verification through..."

        EXAMPLE BAD START:
        "Have you ever wondered about blockchain scalability? 🤔"

        Make each tweet a masterclass in the topic - something people will bookmark and reference.
        """
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error creating tweets: {str(e)}"

    @tool
    def quality_reviewer(tweet_content: str) -> str:
        """Review and enhance tweets to maximize information density and value."""
        prompt = f"""
        Review these tweets and enhance them to be information powerhouses:

        CURRENT TWEETS:
        {tweet_content}

        ENHANCEMENT CRITERIA:

        📊 INFORMATION DENSITY:
        - Are we utilizing every character for maximum value?
        - Can we add more specific technical details?
        - Are there concrete examples or numbers we can include?

        🎯 VALUE PROPOSITION:
        - Will readers learn something genuinely new and useful?
        - Is the technical depth appropriate for crypto Twitter audience?
        - Does each tweet provide actionable insights or understanding?

        🔥 ENGAGEMENT OPTIMIZATION:
        - Does it position readers as early/informed about important developments?
        - Will people want to share this because it makes them look smart?
        - Is there enough substance to spark intelligent discussion?

        ⚡ TWITTER OPTIMIZATION:
        - Use bullet points or numbers for complex information
        - Include relevant technical terms that show expertise
        - Structure for easy reading and shareability

        ENHANCE EACH TWEET BY:
        1. Adding more specific details from the source content
        2. Including concrete examples or use cases
        3. Providing clearer explanations of technical concepts
        4. Maximizing character count with valuable information
        5. Ensuring each tweet can stand alone as valuable content

        Return the enhanced version that transforms casual readers into informed community members.
        """
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error in quality review: {str(e)}"

    @tool
    def kaito_compliance_checker(content: str) -> str:
        """Optimize content for maximum information value and Kaito platform approval."""
        prompt = f"""
        Transform this content into information-rich, Kaito-optimized tweets:

        CURRENT CONTENT:
        {content}

        KAITO PLATFORM OPTIMIZATION:

        🎯 INFORMATION MAXIMIZATION:
        - Pack each tweet with specific, actionable insights
        - Include concrete technical details and explanations
        - Reference exact features, protocols, or capabilities
        - Provide educational value that readers can immediately use

        🔥 CONTENT DEPTH REQUIREMENTS:
        - Each tweet should teach something specific about blockchain/crypto
        - Include "how it works" explanations, not just "what it is"
        - Show the underlying mechanics or technology
        - Explain the "why" behind technical decisions

        📊 KAITO SUCCESS METRICS:
        - High bookmark rate (reference-worthy content)
        - Quote tweets with technical discussions
        - Replies with follow-up questions showing genuine interest
        - Profile visits from people wanting to learn more

        ⚡ TECHNICAL AUTHORITY INDICATORS:
        - Use precise technical terminology correctly
        - Reference specific protocols, algorithms, or implementations
        - Include performance metrics or comparative advantages
        - Demonstrate deep understanding of the technology stack

        🎨 FORMATTING FOR MAXIMUM IMPACT:
        - Use thread format (1/3, 2/3, 3/3) for complex topics
        - Structure with clear takeaways
        - Include relevant emojis for visual breaks
        - End with implications or future developments

        TRANSFORMATION FOCUS:
        - Convert vague statements into specific technical explanations
        - Replace questions with authoritative insights
        - Add concrete examples and use cases
        - Include "under the hood" technical details

        Return content that establishes you as a technical authority while being accessible to the crypto community.
        Each tweet should be dense with valuable, shareable information.
        """
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error in Kaito optimization: {str(e)}"

    @tool
    def final_formatter(content: str) -> str:
        """Format final tweet."""
        return content.strip()

def create_workflow(llm):
    """Create the LangGraph workflow"""
    
    def research_node(state):
        """Enhanced research phase with more comprehensive data gathering"""
        messages = state["messages"]
        query = messages[-1].content
        
        # Gather comprehensive information from all sources
        logger.info(f"Researching query: {query}")
        
        txt_content = text_loader.invoke({"query": query})
        url_content = url_loader.invoke({"query": query})
        tweet_content = tweet_loader.invoke({"query": query})
        
        # Create detailed research summary with more context
        research_summary = f"""
        COMPREHENSIVE RESEARCH FINDINGS FOR QUERY: "{query}"
        
        =================== WHITEPAPER INSIGHTS ===================
        {txt_content}
        
        =================== DOCUMENTATION DETAILS ===================
        {url_content}
        
        =================== COMMUNITY CONTEXT ===================
        {tweet_content}
        
        =================== RESEARCH SYNTHESIS ===================
        The above content represents detailed technical information, documentation, and community insights about Cysic.
        This should be used to create substantial, information-rich content that demonstrates deep technical understanding.
        Focus on extracting specific features, technical innovations, use cases, and unique value propositions.
        """
        
        return {"messages": messages + [AIMessage(content=research_summary)]}

    def synthesis_node(state):
        """Enhanced synthesis with focus on substantial content creation"""
        messages = state["messages"]
        research_data = messages[-1].content
        
        logger.info("Synthesizing research into detailed themes")
        
        themes = content_synthesizer.invoke({"retrieved_content": research_data})
        
        # Add instruction for the next phase
        enhanced_themes = f"""
        DETAILED CONTENT THEMES:
        {themes}
        
        INSTRUCTION FOR TWEET CREATION:
        Use the above detailed themes to create information-dense, educational tweets that showcase deep technical knowledge.
        Each tweet should be packed with specific details, concrete examples, and valuable insights from the research.
        Avoid generic statements and focus on unique technical aspects and real-world applications.
        """
        
        return {"messages": messages + [AIMessage(content=enhanced_themes)]}

    def creation_node(state):
        """Enhanced creation with emphasis on substantial, informative content"""
        messages = state["messages"]
        themes_content = messages[-1].content
        original_query = messages[0].content
        
        logger.info("Creating information-rich tweets")
        
        # Enhanced creation input with clear instructions
        creation_input = f"""
        ORIGINAL USER REQUEST: {original_query}
        
        DETAILED RESEARCH AND THEMES:
        {themes_content}
        
        CREATION GUIDELINES:
        - Create substantial tweets that utilize the rich information from the research
        - Each tweet should be information-dense and educational
        - Include specific technical details, features, and explanations
        - Use concrete examples and real-world applications
        - Demonstrate deep understanding of Cysic's technology
        - Make each tweet a valuable learning resource
        - Avoid generic questions and focus on providing authoritative insights
        """
        
        created_tweets = tweet_creator.invoke({"content": creation_input})
        
        return {"messages": messages + [AIMessage(content=created_tweets)]}

    def quality_node(state):
        """Enhanced quality review with focus on maximizing information value"""
        messages = state["messages"]
        draft_tweet = messages[-1].content
        
        logger.info("Reviewing and enhancing tweet quality")
        
        polished_tweet = quality_reviewer.invoke({"tweet_content": draft_tweet})
        
        return {"messages": messages + [AIMessage(content=polished_tweet)]}

    def kaito_check_node(state):
        """Enhanced Kaito compliance with focus on technical authority"""
        messages = state["messages"]
        polished_tweet = messages[-1].content
        
        logger.info("Optimizing for Kaito platform compliance")
        
        kaito_optimized = kaito_compliance_checker.invoke({"content": polished_tweet})
        
        return {"messages": messages + [AIMessage(content=kaito_optimized)]}

    def final_node(state):
        """Final formatting and preparation for publication"""
        messages = state["messages"]
        kaito_tweet = messages[-1].content
        
        logger.info("Finalizing tweet content")
        
        final_result = final_formatter.invoke({"content": kaito_tweet})
        
        return {"messages": messages + [AIMessage(content=final_result)]}

    # Create workflow
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("research", research_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("creation", creation_node)
    workflow.add_node("quality", quality_node)
    workflow.add_node("kaito_check", kaito_check_node)
    workflow.add_node("final", final_node)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "synthesis")
    workflow.add_edge("synthesis", "creation")
    workflow.add_edge("creation", "quality")
    workflow.add_edge("quality", "kaito_check")
    workflow.add_edge("kaito_check", "final")
    
    return workflow.compile()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Twitter Thread Creator API is running",
        version="1.0.0"
    )

@app.post("/generate-tweets", response_model=TweetResponse)
async def generate_tweets(request: TweetRequest):
    """Generate tweets based on the query"""
    try:
        if not workflow_app:
            raise HTTPException(status_code=500, detail="Workflow not initialized")
        
        logger.info(f"Generating tweets for query: {request.query}")
        
        # Create initial message
        initial_message = HumanMessage(content=request.query)
        
        # Run the workflow
        result = workflow_app.invoke({"messages": [initial_message]})
        
        # Extract the final tweet content
        final_content = result["messages"][-1].content
        
        # Split into individual tweets if multiple
        tweets = []
        if '\n\n' in final_content:
            tweets = [tweet.strip() for tweet in final_content.split('\n\n') if tweet.strip()]
        elif '\n---\n' in final_content:
            tweets = [tweet.strip() for tweet in final_content.split('\n---\n') if tweet.strip()]
        else:
            tweets = [final_content.strip()]
        
        # Clean up tweets and remove any formatting artifacts
        clean_tweets = []
        for tweet in tweets:
            # Remove common thread numbering if present
            clean_tweet = tweet
            if clean_tweet.startswith(('1/', '2/', '3/', '1.', '2.', '3.')):
                lines = clean_tweet.split('\n')
                clean_tweet = '\n'.join(lines[1:]) if len(lines) > 1 else clean_tweet
            clean_tweets.append(clean_tweet.strip())
        
        logger.info(f"Generated {len(clean_tweets)} tweets successfully")
        
        return TweetResponse(
            success=True,
            tweets=clean_tweets,
            metadata={
                "query": request.query,
                "style": request.style,
                "processing_steps": len(result["messages"]) - 1,
                "tweet_count": len(clean_tweets)
            },
            message=f"Successfully generated {len(clean_tweets)} information-rich tweets"
        )
        
    except Exception as e:
        logger.error(f"Error generating tweets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tweets: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Twitter Thread Creator API",
        "version": "1.0.0",
        "description": "Create information-rich Twitter threads using AI and Cysic documentation",
        "endpoints": {
            "health": "/health",
            "generate_tweets": "/generate-tweets",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "RAG-powered content from Cysic whitepaper and documentation",
            "Information-dense tweet generation",
            "Kaito platform optimization",
            "Technical authority positioning",
            "Multi-step quality enhancement"
        ]
    }

# Startup event with proper error handling
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        initialize_components()
        logger.info("✅ Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Application startup failed: {str(e)}")
        raise
    finally:
        # Cleanup (if needed)
        logger.info("Application shutting down")

# Replace the deprecated on_event with lifespan
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variable to suppress warnings
    os.environ.setdefault("USER_AGENT", "TwitterThreadCreator/1.0")
    
    print("🚀 Starting Twitter Thread Creator API...")
    print("📚 Features: RAG-powered, Information-dense, Kaito-optimized")
    print("🔗 Docs: http://localhost:8000/docs")
    print("❤️  Health: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )