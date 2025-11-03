import os
import time
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_groq import ChatGroq

load_dotenv()

# ═══════════════════════════════════════════════════
# OUTPUT CONTROL
# ═══════════════════════════════════════════════════

OUTPUT_INSTRUCTIONS = """
═══════════════════════════════════════════════════
CRITICAL OUTPUT REQUIREMENTS:
═══════════════════════════════════════════════════

YOUR RESPONSE MUST BE:
✓ ONLY the final tweet content - polished and ready to post
✓ NO explanations of your process
✓ NO meta-commentary
✓ NO questions back to the user
✓ NO notes or disclaimers
✓ Just the clean, final tweet

CRITICAL: Every data point, statistic, or claim MUST come directly from the research materials.
If you cannot find specific information in the research, DO NOT invent it.
"""

# ═══════════════════════════════════════════════════
# IMPROVED TEMPLATES WITH RESEARCH GROUNDING
# ═══════════════════════════════════════════════════

TWEET_TEMPLATES = {
    "educational": {
        "name": "Educational/Teaching",
        "description": "Deep educational content that teaches concepts, mechanisms, and principles",
        "best_for": "Building authority, teaching complex topics, data-driven insights",
        "prompt": """
YOUR MISSION: Create an educational crypto tweet about "{topic}" for {project_name} that teaches readers something valuable.

CRITICAL RESEARCH CONSTRAINT:
You MUST use ONLY information from the research materials below. Do NOT invent statistics, features, or claims.
If specific data is not in the research, describe concepts and mechanisms instead of making up numbers.

RESEARCH MATERIALS:
{research_content}

CONTENT STRATEGY:
1. Identify the core mechanism or innovation from the research
2. Explain the "why" behind it - what problem does it solve?
3. Use concrete examples FROM THE RESEARCH (not invented)
4. If the research mentions specific numbers/metrics, use those
5. If no specific metrics exist, focus on explaining concepts clearly
6. Connect to broader crypto principles

EDUCATIONAL APPROACH:
- Open with the key insight or problem
- Explain the mechanism/solution from the research
- Use technical terms correctly (shows expertise)
- Provide the "aha moment" - why this matters
- Close with actionable takeaway

TONE: Expert sharing genuine insight - conversational but credible

LENGTH: {length_spec}

FORMATTING:
- Use *italics* for key concepts
- Use **bold** for critical points (NOT invented statistics)
- Short paragraphs for readability
- Bullets optional for distinct concepts

REMEMBER: Accuracy over impressiveness. Real insights from research beat invented statistics.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "promotional": {
        "name": "Promotional/Announcement",
        "description": "Product launches, feature announcements, updates",
        "best_for": "Marketing, sales, product releases, partnerships",
        "prompt": """
YOUR MISSION: Create compelling promotional content about "{topic}" for {project_name}.

CRITICAL: Use ONLY real information from the research. No invented features or benefits.

RESEARCH MATERIALS:
{research_content}

PROMOTIONAL STRATEGY (using real research):
- Hook: Lead with concrete benefit mentioned in research
- Technical Credibility: Explain actual mechanism from research
- Unique Value: What makes this different based on research?
- Real Proof: Use actual metrics/milestones if mentioned in research
- Call-to-Action: Clear next step

TONE: Excited but credible - hype backed by real substance

LENGTH: {length_spec}

FORMATTING:
- **Bold** real differentiators from research
- Use bullets for actual features (optional)
- Keep it scannable but substantive

Create promotional content grounded in real capabilities, not hype.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "thread": {
        "name": "Twitter Thread",
        "description": "Multi-tweet deep dive (5-8 tweets)",
        "best_for": "Complex topics, comprehensive guides, detailed analyses",
        "prompt": """
YOUR MISSION: Create a Twitter thread (5-8 tweets) about "{topic}" for {project_name}.

CRITICAL: Every claim must be traceable to the research below. No invented data.

RESEARCH MATERIALS:
{research_content}

THREAD STRUCTURE:
Tweet 1 (Hook): Bold claim or insight FROM RESEARCH - make them want to read
Tweet 2-3: Context - Problem/background from research
Tweet 4-6: Deep dive - Explain mechanisms from research with real examples
Tweet 7: Synthesis - Connect the insights
Tweet 8: Takeaway + CTA - Key lesson and engagement ask

Each tweet:
- 200-280 characters
- Based on actual research content
- **Bold** for emphasis (not fake stats)
- Shows expertise through accurate explanation

TONE: Knowledgeable insider sharing real alpha

Separate tweets with:
---TWEET BREAK---

Build credibility through accuracy, not invented drama.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "engagement": {
        "name": "Engagement/Discussion",
        "description": "Questions, hot takes, polls",
        "best_for": "Building community, sparking discussions",
        "prompt": """
YOUR MISSION: Create engagement content about "{topic}" for {project_name}.

RESEARCH MATERIALS:
{research_content}

ENGAGEMENT STRATEGY:
- Base your question/take on real insights from research
- Show you understand the actual mechanisms
- Invite informed discussion
- Use real context from research

TACTICS (pick one):
1. Thought-Provoking Question about real trade-offs
2. Observation about actual mechanisms
3. Ask for perspectives on real features/approaches

STRUCTURE:
- Lead with hook (question/observation)
- Provide context from research
- End with clear invitation to discuss

TONE: Credible insider seeking dialogue

LENGTH: {length_spec}
""" + OUTPUT_INSTRUCTIONS
    },
    
    "casual": {
        "name": "Casual/Observation",
        "description": "Quick thoughts, observations",
        "best_for": "Daily posting, building relatability",
        "prompt": """
YOUR MISSION: Create casual crypto content about "{topic}" for {project_name}.

RESEARCH MATERIALS:
{research_content}

CASUAL STRATEGY:
- Share a genuine observation from the research
- Keep it conversational
- Show understanding naturally (not forced)
- Make it feel like insider knowledge casually shared

TONE: Knowledgeable friend sharing real observations

LENGTH: {length_spec}

Create content that's easy to read and demonstrates real understanding.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "storytelling": {
        "name": "Story/Narrative",
        "description": "Personal stories, case studies, journeys",
        "best_for": "Building connection, sharing lessons",
        "prompt": """
YOUR MISSION: Tell a compelling story about "{topic}" for {project_name}.

RESEARCH MATERIALS:
{research_content}

STORY STRUCTURE:
- Setup: Real context from research
- Challenge: Actual problem/trade-off
- Journey: What happened (from research)
- Resolution: Real outcome
- Lesson: Genuine insight

TONE: Experienced insider sharing real lessons

LENGTH: {length_spec}

Ground your story in real events and insights from the research.
""" + OUTPUT_INSTRUCTIONS
    }
}

# ═══════════════════════════════════════════════════
# USER PROFILE
# ═══════════════════════════════════════════════════

class UserProfile:
    def __init__(
        self,
        user_id: str,
        default_template: str = "educational",
        brand_voice: str = "",
        preferred_length: str = "medium",
    ):
        self.user_id = user_id
        self.default_template = default_template
        self.brand_voice = brand_voice
        self.preferred_length = preferred_length

# ═══════════════════════════════════════════════════
# CPU-OPTIMIZED MAIN CONTENT CREATOR CLASS
# ═══════════════════════════════════════════════════

class MultiUserContentCreator:
    def __init__(self, API_KEY: str):
        if not API_KEY:
            raise ValueError("GROQ API key is required")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            api_key=API_KEY, 
            temperature=0.4
        )

        print("Initializing CPU-optimized embeddings...")
        
        # ═══ CPU OPTIMIZATION 1: Batch processing ═══
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,  # Process 64 chunks at once - 50% faster!
                "show_progress_bar": False
            }
        )

        # ═══ CPU OPTIMIZATION 2: Smaller chunks = fewer embeddings ═══
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,      # Reduced from 1000 - 25% fewer chunks
            chunk_overlap=100,   # Reduced from 200 - less redundancy
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.user_profiles: Dict[str, UserProfile] = {}
        self.vector_store: Dict[str, Chroma] = {}
        self.project_name: Dict[str, str] = {}
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def register_user(self, user_id: str, **preferences):
        """Register a new user with optional preferences."""
        profile = UserProfile(user_id, **preferences)
        self.user_profiles[user_id] = profile
        print(f"Successfully registered user: {user_id}")
        print(f"Default template: {profile.default_template}")
        if profile.brand_voice:
            print(f"   Brand voice: {profile.brand_voice}")
        return profile

    def get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile, creating default if doesn't exist."""
        if user_id not in self.user_profiles:
            print(f"No user id found for: {user_id}. Creating default profile.")
            self.register_user(user_id)
        return self.user_profiles[user_id]

    def list_available_templates(self):
        """Show all available tweet templates."""
        print("\nAVAILABLE TEMPLATES:\n")
        for key, template in TWEET_TEMPLATES.items():
            print(f"{key.upper()}: {template['name']}")
            print(f"Description: {template['description']}")
            print(f"Best for: {template['best_for']}\n")

    def _get_project_path(self, user_id: str, project_name: str) -> str:
        """Get the persist directory path for a project."""
        return os.path.join(self.data_dir, f"{user_id}_{project_name.lower()}")

    def _save_project_metadata(self, user_id: str, project_name: str, urls: List[str], whitepaper_path: Optional[str]):
        """Save project metadata."""
        persist_dir = self._get_project_path(user_id, project_name)
        metadata = {
            "project_name": project_name,
            "user_id": user_id,
            "urls": urls,
            "whitepaper_path": whitepaper_path,
            "created_at": time.time()
        }
        with open(os.path.join(persist_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_project_metadata(self, user_id: str, project_name: str) -> Optional[Dict]:
        """Load project metadata."""
        persist_dir = self._get_project_path(user_id, project_name)
        metadata_path = os.path.join(persist_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    def list_user_projects(self, user_id: str) -> List[Dict]:
        """List all projects for a user."""
        projects = []
        
        if not os.path.exists(self.data_dir):
            return projects
        
        for dir_name in os.listdir(self.data_dir):
            if dir_name.startswith(f"{user_id}_"):
                project_name = dir_name.replace(f"{user_id}_", "")
                metadata = self._load_project_metadata(user_id, project_name)
                
                if metadata:
                    projects.append({
                        "project_name": project_name,
                        "created_at": metadata.get("created_at"),
                        "urls_count": len(metadata.get("urls", [])),
                        "has_whitepaper": bool(metadata.get("whitepaper_path"))
                    })
        
        return projects

    def load_project(self, user_id: str, project_name: str) -> bool:
        """Load an existing project from disk."""
        persist_dir = self._get_project_path(user_id, project_name)
        
        if not os.path.exists(persist_dir):
            print(f"Project '{project_name}' not found for user {user_id}")
            return False
        
        try:
            print(f"Loading project '{project_name}' for user {user_id}...")
            
            store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
                collection_name=f"{user_id}_{project_name.lower()}"
            )
            
            self.vector_store[user_id] = store
            self.project_name[user_id] = project_name
            
            metadata = self._load_project_metadata(user_id, project_name)
            if metadata:
                print(f"Loaded project with {len(metadata.get('urls', []))} sources")
            
            return True
            
        except Exception as e:
            print(f"Error loading project: {e}")
            return False

    # ═══════════════════════════════════════════════════
    # CPU-OPTIMIZED SETUP PROJECT
    # ═══════════════════════════════════════════════════
    
    def setup_project(self, user_id: str, project_name: str, urls: List[str], whitepaper_path: Optional[str] = None):
        """Create a new project or update existing one - CPU OPTIMIZED VERSION."""
        start_time = time.time()
        
        profile = self.get_user_profile(user_id)
        persist_dir = self._get_project_path(user_id, project_name)
        
        print(f"\n{'='*60}")
        print(f"Setting up project '{project_name}' for user: {user_id}")
        print(f"{'='*60}")
        all_documents = []
        
        # ═══ CPU OPTIMIZATION 3: URL loading with timeout ═══
        url_start = time.time()
        if urls:
            print(f"\n📥 Loading {len(urls)} URLs...")
            for i, url in enumerate(urls, 1):
                try:
                    url_load_start = time.time()
                    loader = WebBaseLoader([url])
                    loader.requests_kwargs = {'timeout': 15}  # Prevent hanging
                    docs = loader.load()
                    chunks = self.text_splitter.split_documents(docs)
                    all_documents.extend(chunks)
                    elapsed = time.time() - url_load_start
                    print(f"   ✓ {i}/{len(urls)}: {len(chunks)} chunks ({elapsed:.1f}s) - {url[:50]}")
                except Exception as e:
                    print(f"   ✗ {i}/{len(urls)}: Failed - {str(e)[:60]}")
            
            print(f"\n⏱️  URL Loading: {time.time() - url_start:.1f}s")
        
        # ═══ WHITEPAPER PROCESSING ═══
        if whitepaper_path and os.path.exists(whitepaper_path):
            wp_start = time.time()
            print(f"\n📄 Processing whitepaper: {whitepaper_path}")
            try:
                loader = TextLoader(whitepaper_path)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                all_documents.extend(chunks)
                print(f"   ✓ Added {len(chunks)} chunks ({time.time() - wp_start:.1f}s)")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        
        if not all_documents:
            return f"❌ No documents found for {project_name}."
        
        os.makedirs(persist_dir, exist_ok=True)
        
        # ═══ CPU OPTIMIZATION 4: Efficient vector store creation ═══
        vector_start = time.time()
        total_chunks = len(all_documents)
        print(f"\n🧠 Creating embeddings for {total_chunks} chunks...")
        print(f"   Estimated time: {total_chunks * 0.04:.0f}-{total_chunks * 0.08:.0f} seconds")
        
        try:
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                # Update existing project
                print("   Loading existing vector store...")
                store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                    collection_name=f"{user_id}_{project_name.lower()}"
                )
                
                # ═══ CPU OPTIMIZATION 5: Batch additions ═══
                batch_size = 100
                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i+batch_size]
                    batch_start = time.time()
                    store.add_documents(batch)
                    batch_time = time.time() - batch_start
                    progress = min(i + batch_size, total_chunks)
                    print(f"   Progress: {progress}/{total_chunks} chunks ({batch_time:.1f}s)")
            else:
                # Create new vector store
                print("   Creating new vector store...")
                store = Chroma.from_documents(
                    documents=all_documents,
                    embedding=self.embeddings,
                    collection_name=f"{user_id}_{project_name.lower()}",
                    persist_directory=persist_dir,
                )
                print(f"   ✓ Created with {total_chunks} chunks")
            
            embed_time = time.time() - vector_start
            print(f"\n⏱️  Embedding Creation: {embed_time:.1f}s")
            print(f"   Average: {embed_time/total_chunks:.3f}s per chunk")
            
        except Exception as e:
            print(f"\n❌ Vector store creation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error creating vector store: {e}"
        
        self.vector_store[user_id] = store
        self.project_name[user_id] = project_name
        
        self._save_project_metadata(user_id, project_name, urls, whitepaper_path)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ PROJECT SETUP COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Total chunks: {total_chunks}")
        print(f"Average: {total_time/total_chunks:.3f}s per chunk")
        print(f"{'='*60}\n")
        
        return f"Project '{project_name}' setup complete with {total_chunks} chunks."
                            
    def delete_project(self, user_id: str, project_name: str) -> bool:
        """Delete a project and all its data."""
        persist_dir = self._get_project_path(user_id, project_name)
        
        if not os.path.exists(persist_dir):
            print(f"Project '{project_name}' not found")
            return False
        
        try:
            import shutil
            shutil.rmtree(persist_dir)
            
            if user_id in self.vector_store and self.project_name.get(user_id) == project_name:
                del self.vector_store[user_id]
                del self.project_name[user_id]
            
            print(f"Project '{project_name}' deleted")
            return True
            
        except Exception as e:
            print(f"Error deleting project: {e}")
            return False

    def create_twitter_content(
        self, 
        user_id: str, 
        topic: str, 
        template: Optional[str] = None,
        length: str = "medium",
        debug: bool = False
    ) -> str:
        """Create Twitter content using specified template."""
        
        # Auto-load project if needed
        if user_id not in self.vector_store:
            projects = self.list_user_projects(user_id)
            if projects:
                first_project = projects[0]["project_name"]
                print(f"Auto-loading project: {first_project}")
                if not self.load_project(user_id, first_project):
                    return "Error: No project loaded. Please run setup_project() first."
            else:
                return "Error: No projects found. Run setup_project() first."
        
        profile = self.get_user_profile(user_id)
        template_key = template or profile.default_template
        
        if template_key not in TWEET_TEMPLATES:
            available = ", ".join(TWEET_TEMPLATES.keys())
            return f"Error: Invalid template '{template_key}'. Available: {available}"
        
        template_config = TWEET_TEMPLATES[template_key]
        
        # Retrieve relevant documents
        store = self.vector_store[user_id]
        k = 8 if template_key in ["educational", "thread"] else 5
        
        docs = store.similarity_search(topic, k=k)
        
        if not docs:
            return f"Error: No relevant documents found for topic '{topic}'. Check your project setup."
        
        research_content = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Debug mode
        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG - Retrieved {len(docs)} documents")
            print(f"Total research content length: {len(research_content)} chars")
            print(f"{'='*60}")
            print("First document preview:")
            print(docs[0].page_content[:500])
            print(f"{'='*60}\n")
        
        # Check research quality
        if len(research_content) < 200:
            print("WARNING: Very little research content found.")
            print(f"   Only {len(research_content)} characters retrieved.")
            print("   Results may be generic. Consider:")
            print("   1. Adding more relevant documentation")
            print("   2. Checking if URLs loaded correctly")
            print("   3. Using a more specific topic query")
        
        # Length specs
        length_specs = {
            "short": "400-600 characters",
            "medium": "600-850 characters",
            "long": "900-1500 characters"
        }
        
        # Build prompt
        prompt = template_config["prompt"].format(
            topic=topic,
            project_name=self.project_name.get(user_id, "your project"),
            research_content=research_content[:6000],
            length_spec=length_specs.get(length, length_specs["medium"])
        )
        
        if profile.brand_voice:
            prompt += f"\n\nBRAND VOICE: {profile.brand_voice}"
        
        print(f"\n🎯 Creating {template_key} content about: {topic} ({length})")
        
        system_message = SystemMessage(content="""You are a professional crypto Twitter content creator.

CRITICAL RULES:
1. Use ONLY information from the research materials provided
2. Do NOT invent statistics, features, or claims
3. If you cannot find specific data, focus on explaining concepts accurately
4. Output ONLY the final tweet - no explanations, no meta-commentary
5. Every data point must be traceable to the research

You create accurate, insightful content grounded in real information.""")
        
        response = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        return response.content

    def tweak_content(self, user_id: str, original_content: str, feedback: str) -> str:
        """Iterate on existing content based on feedback."""
        
        system_message = SystemMessage(content="""You are a professional crypto Twitter content creator.
You revise content based on feedback. Output ONLY the revised tweet - no explanations.""")
        
        prompt = f"""
Original content:
{original_content}

User feedback:
{feedback}

Rewrite based on feedback while maintaining accuracy. Output ONLY the revised tweet.
"""
        response = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        return response.content


def create_content_system(groq_api_key: str):
    return MultiUserContentCreator(groq_api_key)
