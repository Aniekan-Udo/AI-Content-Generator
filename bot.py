import os
import pprint
from dotenv import load_dotenv
load_dotenv()
import operator
from typing import Annotated
from IPython.display import Image, display

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader,PyPDFLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.tools import TavilySearchResults
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import operator
from langchain.docstore.document import Document
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langgraph_supervisor import create_supervisor
from IPython.display import Image, display

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document

llm = ChatGroq(model="llama-3.3-70b-versatile", GROQ_API_KEY="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")

import tweepy

client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAACsy4QEAAAAAl%2F6hXVFuAw1ih2GBPjR%2BHJQkxZI%3DGeVeQJ2AZNU9HTbE1ajiwaVAvaUIrtCfRO9jE7hOEk1Ybb6Gj0")

# Step 1: Get user ID for the handle
user = client.get_user(username="Cysic_xyz")
user_id = user.data.id

# Step 2: Get tweets
tweets = client.get_users_tweets(
    id=user_id,
    max_results=10,  # latest 10 tweets
    tweet_fields=["created_at", "text"]
)

for tweet in tweets.data:
    print(f"[{tweet.created_at}] {tweet.text}")


# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Local TXT Retriever (Enhanced for deeper content) ---
doc_loader = TextLoader(r"C:\Users\HP\Desktop\Twitter Thread Creator\Cysic whitepaper.txt")
docs = doc_loader.load()
# Larger chunks for more comprehensive content
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
docs_chunk = text_splitter.split_documents(docs)

pdf_store = Chroma.from_documents(
    documents=docs_chunk,
    collection_name="text_docs",
    embedding=embeddings,
    persist_directory="./chroma_store_txt"
)
# Retrieve more chunks for comprehensive threads
txt_retriever = pdf_store.as_retriever(search_kwargs={"k": 8})

# --- URL Retriever (Enhanced for deeper content) ---
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
url_retriever = url_store.as_retriever(search_kwargs={"k": 8})

# --- Tweet Retriever (Enhanced filtering for meaningful content) ---
def filter_meaningful_tweets(tweets):
    """Filter and process tweets to store only meaningful, generic content"""
    if not tweets:  # Handle empty list
        return []
        
    meaningful_tweets = []
    
    for tweet in tweets:
        # Ensure tweet is a string
        if not isinstance(tweet, str):
            continue
            
        # Skip if too short (less than 10 characters to be more lenient)
        if len(tweet.strip()) < 10:
            continue
            
        # Skip if contains too many mentions or hashtags (spam-like)
        mention_count = tweet.count('@')
        hashtag_count = tweet.count('#')
        if mention_count > 5 or hashtag_count > 8:  # More lenient
            continue
            
        # Skip promotional/spam content
        spam_keywords = ['buy now', 'limited time', 'click here', 'dm me', 'follow for follow', 'get rich quick']
        if any(keyword in tweet.lower() for keyword in spam_keywords):
            continue
            
        # Keep educational, informational, or engaging content
        meaningful_keywords = ['learn', 'understand', 'explore', 'discover', 'innovation', 
                             'technology', 'development', 'community', 'future', 'breakthrough',
                             'cysic', 'blockchain', 'crypto', 'defi', 'web3', 'protocol']
        
        # More lenient criteria: keep if has meaningful keywords OR is reasonably long OR mentions project
        if (any(keyword in tweet.lower() for keyword in meaningful_keywords) or 
            len(tweet) > 80 or 
            'cysic' in tweet.lower()):
            meaningful_tweets.append(tweet.strip())
    
    return meaningful_tweets

# Process Tweepy response object properly
def extract_tweet_texts(tweets_response):
    """Extract tweet texts from Tweepy response object"""
    if not tweets_response or not hasattr(tweets_response, 'data') or not tweets_response.data:
        return []
    
    tweet_texts = []
    for tweet in tweets_response.data:
        if hasattr(tweet, 'text'):
            tweet_texts.append(tweet.text)
    
    return tweet_texts

# Check if 'tweets' variable exists and handle appropriately
try:
    if 'tweets' in locals() or 'tweets' in globals():
        # Extract text from Tweepy response
        raw_tweet_texts = extract_tweet_texts(tweets)
        filtered_tweets = filter_meaningful_tweets(raw_tweet_texts)
        print(f"Original tweets: {len(raw_tweet_texts)}")
        print(f"Filtered tweets: {len(filtered_tweets)}")
        
        # Print sample of filtered tweets for debugging
        if filtered_tweets:
            print("\nSample filtered tweets:")
            for i, tweet in enumerate(filtered_tweets[:3], 1):
                print(f"{i}. {tweet[:100]}...")
                
    elif 'tweet' in locals() or 'tweet' in globals():
        # Handle if you named it 'tweet' instead of 'tweets'
        if hasattr(tweet, 'data'):
            raw_tweet_texts = extract_tweet_texts(tweet)
            filtered_tweets = filter_meaningful_tweets(raw_tweet_texts)
        else:
            filtered_tweets = filter_meaningful_tweets(tweet)
        print(f"Filtered tweets: {len(filtered_tweets)}")
        
    else:
        # If no tweets available, create some sample meaningful tweets for testing
        print("No 'tweets' variable found. Using sample tweets for testing.")
        sample_tweets = [
            "Exploring the future of decentralized protocols with Cysic's innovative approach to blockchain technology.",
            "The community is growing! Excited to see more developers joining the Cysic ecosystem.",
            "Understanding zero-knowledge proofs and their role in modern cryptocurrency systems.",
            "Innovation in blockchain requires both technical excellence and community collaboration.",
            "Building the future of Web3 infrastructure one protocol at a time."
        ]
        filtered_tweets = sample_tweets
        
except Exception as e:
    print(f"Error processing tweets: {e}")
    # Fallback to sample tweets
    filtered_tweets = [
        "Cysic is revolutionizing blockchain technology through innovative protocols.",
        "Join our growing community of developers building the future of DeFi.",
        "Learn about the latest developments in zero-knowledge proof systems."
    ]

# Only create tweet store if we have tweets
if filtered_tweets:
    tweet_docs = [Document(page_content=t, metadata={"source": "twitter", "type": "meaningful_content"}) 
                  for t in filtered_tweets]

    tweet_store = Chroma.from_documents(
        embedding=embeddings,
        collection_name="tweet_docs",
        documents=tweet_docs,
        persist_directory="./chroma_store_tweet"
    )
    tweet_retriever = tweet_store.as_retriever(search_kwargs={"k": 5})
else:
    print("Warning: No meaningful tweets found. Tweet retriever will return empty results.")
    # Create a dummy retriever that returns empty results
    class EmptyRetriever:
        def get_relevant_documents(self, query):
            return []
    
    tweet_retriever = EmptyRetriever()

# -------------------------
# ENHANCED TOOLS FOR EDUCATIONAL THREAD CREATION
# -------------------------
MAX_CONTENT_CHARS = 6000  # Increased for comprehensive threads

@tool
def comprehensive_research_tool(query: str) -> str:
    """Comprehensive research tool that gathers extensive content from all sources for educational threads."""
    
    # Get comprehensive content from all sources
    txt_results = txt_retriever.get_relevant_documents(query)
    url_results = url_retriever.get_relevant_documents(query)
    
    try:
        tweet_results = tweet_retriever.get_relevant_documents(query)
    except:
        tweet_results = []
    
    # Compile comprehensive research
    research_data = {
        "whitepaper_content": [],
        "documentation_content": [],
        "community_insights": [],
        "total_sources": 0
    }
    
    # Process whitepaper content
    for doc in txt_results:
        research_data["whitepaper_content"].append({
            "content": doc.page_content,
            "relevance": "high"
        })
        research_data["total_sources"] += 1
    
    # Process documentation content
    for doc in url_results:
        research_data["documentation_content"].append({
            "content": doc.page_content,
            "relevance": "high"
        })
        research_data["total_sources"] += 1
    
    # Process community insights
    for doc in tweet_results:
        research_data["community_insights"].append({
            "content": doc.page_content,
            "relevance": "medium"
        })
        research_data["total_sources"] += 1
    
    # Compile into comprehensive research summary
    compiled_research = f"""
    COMPREHENSIVE RESEARCH FOR: {query}
    
    TOTAL SOURCES ANALYZED: {research_data["total_sources"]}
    
    === WHITEPAPER INSIGHTS ===
    {chr(10).join([f"• {item['content']}" for item in research_data["whitepaper_content"]])}
    
    === TECHNICAL DOCUMENTATION ===
    {chr(10).join([f"• {item['content']}" for item in research_data["documentation_content"]])}
    
    === COMMUNITY CONTEXT ===
    {chr(10).join([f"• {item['content']}" for item in research_data["community_insights"]])}
    """
    
    return compiled_research[:MAX_CONTENT_CHARS]

@tool
def educational_thread_architect(research_content: str, thread_length: int = 5) -> str:
    """Architect comprehensive educational Twitter threads with deep, valuable content."""
    
    architect_prompt = f"""
    You are an expert Twitter thread architect specializing in EDUCATIONAL content that teaches and informs.

    Based on this comprehensive research, create a detailed thread structure:
    {research_content}

    THREAD REQUIREMENTS:
    - Length: {thread_length} tweets minimum
    - Educational focus: Teach complex concepts simply
    - Value-driven: Each tweet must provide substantial value
    - Progressive structure: Build knowledge step by step
    - Include specific technical details and examples
    - Make complex topics accessible

    THREAD ARCHITECTURE FRAMEWORK:

    🧵 TWEET 1 - HOOK & CONTEXT:
    - Start with a compelling statement about the topic's importance
    - Provide context for why this matters to the crypto/tech community
    - Preview the key insights they'll learn

    🧵 TWEET 2-3 - FOUNDATIONAL CONCEPTS:
    - Explain the core technical concepts in accessible terms
    - Use analogies and examples from the research
    - Build the foundation for deeper understanding

    🧵 TWEETS 4-5+ - DEEP DIVE & TECHNICAL DETAILS:
    - Present specific technical innovations and mechanisms
    - Include concrete examples and use cases
    - Show real-world applications and implications

    🧵 FINAL TWEET - SYNTHESIS & IMPACT:
    - Synthesize key learnings
    - Explain broader implications for the industry
    - End with thought-provoking insights or future outlook

    CONTENT GUIDELINES:
    ✅ Use specific data, facts, and technical details from the research
    ✅ Explain complex concepts with clear, simple language
    ✅ Include concrete examples and real-world applications  
    ✅ Build a logical, progressive narrative
    ✅ Each tweet should teach something new and valuable
    ✅ Focus on "how" and "why" rather than just "what"
    ✅ Include technical depth that demonstrates expertise

    ❌ Avoid vague generalizations
    ❌ Don't just ask questions - provide answers and insights
    ❌ Avoid fluff or filler content
    ❌ Don't repeat the same points across tweets

    Return a detailed thread structure with specific content points for each tweet.
    """
    
    response = llm.invoke(architect_prompt)
    return response.content

@tool
def thread_content_creator(thread_structure: str, tone: str = "educational") -> str:
    """Create the actual Twitter thread content based on the architectural structure."""
    
    creator_prompt = f"""
    You are an expert Twitter content creator specializing in educational threads.

    Transform this thread structure into actual, ready-to-post Twitter content:
    {thread_structure}

    CONTENT CREATION GUIDELINES:

    📝 WRITING STYLE:
    - {tone} tone that's authoritative yet accessible
    - Use clear, concise language that explains complex concepts
    - Include specific technical details and examples
    - Make each tweet substantive and valuable
    - Use engaging hooks and smooth transitions

    🎯 TWEET FORMATTING:
    - Each tweet should be 200-280 characters for optimal engagement
    - Use thread numbering (1/, 2/, 3/, etc.)
    - Include relevant emojis for visual appeal and organization
    - Use line breaks for readability
    - End each tweet (except the last) with a hook for the next tweet

    📊 CONTENT REQUIREMENTS:
    - Include specific data points, technical details, and examples
    - Explain mechanisms, processes, and innovations clearly
    - Use analogies to make complex concepts accessible
    - Provide actionable insights and key takeaways
    - Focus on teaching and educating the audience

    🔗 THREAD FLOW:
    - Ensure smooth transitions between tweets
    - Build knowledge progressively
    - Maintain engagement throughout the entire thread
    - End with a strong conclusion that synthesizes key points

    Return the complete, ready-to-post Twitter thread with proper formatting.
    """
    
    response = llm.invoke(creator_prompt)
    return response.content

@tool
def thread_quality_enhancer(thread_content: str) -> str:
    """Enhance thread quality for maximum educational value and engagement."""
    
    enhancer_prompt = f"""
    You are a Twitter thread optimization expert focused on educational content.

    Review and enhance this thread for maximum educational value and engagement:
    {thread_content}

    ENHANCEMENT CRITERIA:

    📚 EDUCATIONAL VALUE:
    - Does each tweet teach something concrete and valuable?
    - Are complex concepts explained clearly and simply?
    - Is there sufficient technical depth to demonstrate expertise?
    - Are there specific examples and use cases included?

    🔥 ENGAGEMENT OPTIMIZATION:
    - Strong, attention-grabbing opening hook
    - Compelling progression that keeps readers engaged
    - Clear value proposition in each tweet
    - Thought-provoking insights and implications
    - Strong conclusion that synthesizes learnings

    ✨ CONTENT QUALITY:
    - Remove any vague or generic statements
    - Strengthen weak transitions between tweets
    - Add specific technical details where missing
    - Improve clarity and accessibility of explanations
    - Ensure each tweet provides substantial value

    📱 TWITTER OPTIMIZATION:
    - Optimize tweet length for readability
    - Improve formatting and visual appeal
    - Ensure proper thread numbering and flow
    - Add strategic emojis for organization and appeal

    🎯 KAITO COMPLIANCE:
    - High educational and informational value
    - Original insights and perspectives
    - Technical depth and expertise demonstration
    - Community-relevant content
    - Discussion-worthy angles

    Return the enhanced, optimized thread ready for publication.
    """
    
    response = llm.invoke(enhancer_prompt)
    return response.content

@tool
def thread_formatter(enhanced_thread: str) -> str:
    """Final formatting for publication-ready Twitter thread."""
    
    formatter_prompt = f"""
    Apply final formatting to this Twitter thread for publication:
    {enhanced_thread}

    FINAL FORMATTING REQUIREMENTS:

    📱 TWITTER SPECIFICATIONS:
    - Ensure each tweet is under 280 characters
    - Use proper thread numbering (1/X, 2/X, etc.)
    - Add strategic line breaks for readability
    - Include relevant emojis for visual organization
    - Ensure smooth flow between tweets

    🎨 VISUAL OPTIMIZATION:
    - Use consistent formatting throughout
    - Strategic use of emojis and symbols
    - Proper spacing and line breaks
    - Clear section divisions
    - Easy-to-scan structure

    ✅ FINAL CHECKLIST:
    - Each tweet provides substantial value
    - Technical concepts are clearly explained
    - Specific examples and details are included
    - Thread builds knowledge progressively
    - Strong opening and closing
    - Ready to copy-paste and publish

    Return the final, publication-ready Twitter thread.
    """
    
    response = llm.invoke(formatter_prompt)
    return response.content

# -------------------------
# ENHANCED WORKFLOW FOR EDUCATIONAL THREADS
# -------------------------

from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

def comprehensive_research_node(state):
    """Enhanced research phase: gather extensive content from all sources"""
    messages = state["messages"]
    query = messages[-1].content
    
    # Extract thread length if specified
    thread_length = 5  # default
    if "thread" in query.lower():
        # Try to extract number
        import re
        numbers = re.findall(r'\d+', query)
        if numbers:
            thread_length = min(int(numbers[0]), 15)  # Max 15 tweets
    
    # Comprehensive research
    research_content = comprehensive_research_tool.invoke({"query": query})
    
    research_summary = f"""
    COMPREHENSIVE RESEARCH COMPLETED
    Query: {query}
    Requested Thread Length: {thread_length}
    
    {research_content}
    """
    
    return {"messages": messages + [AIMessage(content=research_summary)]}

def thread_architecture_node(state):
    """Architect the educational thread structure"""
    messages = state["messages"]
    research_data = messages[-1].content
    
    # Extract thread length from research summary
    thread_length = 5
    if "Thread Length:" in research_data:
        import re
        match = re.search(r'Thread Length: (\d+)', research_data)
        if match:
            thread_length = int(match.group(1))
    
    # Create thread architecture
    thread_structure = educational_thread_architect.invoke({
        "research_content": research_data,
        "thread_length": thread_length
    })
    
    return {"messages": messages + [AIMessage(content=thread_structure)]}

def content_creation_node(state):
    """Create the actual thread content"""
    messages = state["messages"]
    thread_structure = messages[-1].content
    original_query = messages[0].content
    
    # Determine tone from original query
    tone = "educational"
    if "technical" in original_query.lower():
        tone = "technical-educational"
    elif "beginner" in original_query.lower():
        tone = "beginner-friendly"
    
    # Create thread content
    thread_content = thread_content_creator.invoke({
        "thread_structure": thread_structure,
        "tone": tone
    })
    
    return {"messages": messages + [AIMessage(content=thread_content)]}

def quality_enhancement_node(state):
    """Enhance thread quality and educational value"""
    messages = state["messages"]
    thread_content = messages[-1].content
    
    # Enhance the thread
    enhanced_thread = thread_quality_enhancer.invoke({"thread_content": thread_content})
    
    return {"messages": messages + [AIMessage(content=enhanced_thread)]}

def final_formatting_node(state):
    """Final formatting for publication"""
    messages = state["messages"]
    enhanced_thread = messages[-1].content
    
    # Final formatting
    final_thread = thread_formatter.invoke({"enhanced_thread": enhanced_thread})
    
    return {"messages": messages + [AIMessage(content=final_thread)]}

# Create the enhanced workflow
workflow = StateGraph(MessagesState)

# Add nodes for educational thread creation
workflow.add_node("research", comprehensive_research_node)
workflow.add_node("architect", thread_architecture_node)
workflow.add_node("create", content_creation_node)
workflow.add_node("enhance", quality_enhancement_node)
workflow.add_node("format", final_formatting_node)

# Define the flow for educational thread generation
workflow.set_entry_point("research")
workflow.add_edge("research", "architect")
workflow.add_edge("architect", "create")
workflow.add_edge("create", "enhance")
workflow.add_edge("enhance", "format")

# Compile the enhanced application
app = workflow.compile()

print("✅ Enhanced Educational Thread Creator loaded successfully!")
print("🎯 Optimized for comprehensive, educational Twitter threads")
print("📚 Uses deep content from vector stores for substantial value")