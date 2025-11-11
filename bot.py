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
# ENHANCED TEMPLATES WITH RELATABILITY
# ═══════════════════════════════════════════════════

TWEET_TEMPLATES = {
    "educational": {
        "name": "Educational/Teaching",
        "description": "Deep educational content that teaches concepts, mechanisms, and principles",
        "best_for": "Building authority, teaching complex topics, data-driven insights",
        "prompt": """
YOUR MISSION: Create an educational crypto tweet about "{topic}" for {project_name} that teaches readers something valuable WHILE connecting with them personally.

CRITICAL RESEARCH CONSTRAINT:
You MUST use ONLY information from the research materials below. Do NOT invent statistics, features, or claims.
If specific data is not in the research, describe concepts and mechanisms instead of making up numbers.

RESEARCH MATERIALS:
{research_content}

RELATABILITY LAYER (Apply to technical content):
- Open with a relatable problem/frustration readers face
- Use "you" language: "You've probably noticed..." "Ever wondered why..."
- Include 1 analogy to everyday life (e.g., "Think of it like...")
- Acknowledge common misconceptions or confusions
- Make technical concepts feel accessible, not intimidating

CONTENT STRATEGY:
1. Hook with relatable pain point or curiosity gap
2. Identify the core mechanism/innovation from research
3. Explain using accessible language + one real-world analogy
4. Use concrete examples FROM THE RESEARCH (not invented)
5. Connect to what readers care about (their money, time, security)
6. Close with "aha moment" that feels personally relevant

EDUCATIONAL APPROACH:
- Open with empathy: "Tired of X?" or "You know that feeling when..."
- Explain the "why" behind it in human terms
- Use technical terms but define them conversationally
- Provide the "aha moment" that connects to their experience
- Close with actionable takeaway they can use

TONE: Smart friend explaining something cool over coffee - knowledgeable but never condescending

LENGTH: {length_spec}

FORMATTING:
- Use *italics* for key concepts
- Use **bold** for critical points (NOT invented statistics)
- Short paragraphs for readability
- Bullets optional for distinct concepts

REMEMBER: Accuracy + Connection. Teach real insights while making readers feel understood.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "promotional": {
        "name": "Promotional/Announcement",
        "description": "Product launches, feature announcements, updates",
        "best_for": "Marketing, sales, product releases, partnerships",
        "prompt": """
YOUR MISSION: Create compelling promotional content about "{topic}" for {project_name} that connects emotionally while showcasing real value.

CRITICAL: Use ONLY real information from the research. No invented features or benefits.

RESEARCH MATERIALS:
{research_content}

RELATABILITY FIRST:
- Lead with the FEELING: "Imagine never worrying about X again..."
- Show you understand their current pain
- Use "you" and "your" extensively
- Paint a picture of their improved reality

PROMOTIONAL STRATEGY (using real research):
- Hook: Lead with emotional benefit, then concrete feature from research
- Empathy: "We know [pain point]..." 
- Solution: Explain actual mechanism from research in accessible terms
- Unique Value: What makes this different (based on research) FOR THEM
- Real Proof: Use actual metrics/milestones if mentioned
- Call-to-Action: Clear next step that benefits THEM

TONE: Excited friend sharing something that genuinely helps - hype backed by empathy and real substance

LENGTH: {length_spec}

FORMATTING:
- **Bold** real differentiators from research
- Use bullets for actual features (optional)
- Keep it scannable but emotionally engaging

Create promotional content that makes people FEEL something, then backs it with real capabilities.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "thread": {
        "name": "Twitter Thread",
        "description": "Multi-tweet deep dive (5-8 tweets)",
        "best_for": "Complex topics, comprehensive guides, detailed analyses",
        "prompt": """
YOUR MISSION: Create a RELATABLE Twitter thread (5-8 tweets) about "{topic}" for {project_name} that educates AND connects.

CRITICAL: Every claim must be traceable to the research below. No invented data.

RESEARCH MATERIALS:
{research_content}

THREAD STRUCTURE WITH RELATABILITY:

Tweet 1 (Relatable Hook): 
- Start with shared experience or frustration
- "Ever felt...?" "You know that moment when...?" "Tired of...?"
- Make them think "YES, that's me!"

Tweet 2 (Empathy + Context):
- "Here's why that happens..." 
- Validate their experience
- Set up the problem from research with emotional resonance

Tweet 3-4 (Story + Explanation):
- Use mini-narrative or "imagine this..." scenario
- Explain mechanism from research through relatable lens
- Include everyday analogy: "It's like when you..."
- Ground in research but make it feel personal

Tweet 5-6 (Deep Dive with Connection):
- Technical details explained conversationally
- "Think about it this way..."
- Real examples from research presented as "picture this..."
- Keep connecting back to reader's experience

Tweet 7 (Synthesis with Emotion):
- "This is why it matters to YOU"
- Connect insights to their goals/fears/desires
- Make the technical personal

Tweet 8 (Takeaway + Community CTA):
- Key lesson in memorable, human language
- Invite discussion: "What's your experience with...?"
- Make them feel part of a community

Each tweet:
- 200-280 characters
- Based on actual research content
- Use conversational language: contractions, casual phrasing
- **Bold** for emphasis (not fake stats)
- "You" language throughout

TONE: Knowledgeable friend sharing insights that genuinely help - like explaining something important to someone you care about

RELATABILITY TACTICS:
- Personal pronouns: "you", "your", "we"
- Rhetorical questions: "Sound familiar?"
- Shared frustrations: "We've all been there"
- Mini-stories or scenarios readers can visualize
- Emotional words: frustrated, excited, worried, relieved
- Casual language: "here's the thing", "honestly", "real talk"

Separate tweets with:
---TWEET BREAK---

Build credibility through accuracy AND emotional intelligence. Teach while making readers feel seen.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "engagement": {
        "name": "Engagement/Discussion",
        "description": "Questions, hot takes, polls",
        "best_for": "Building community, sparking discussions",
        "prompt": """
YOUR MISSION: Create engagement content about "{topic}" for {project_name} that invites genuine conversation and connection.

RESEARCH MATERIALS:
{research_content}

RELATABILITY FOCUS:
- Make it about THEIR experience, not just the tech
- Use vulnerable or honest language
- Ask questions that tap into emotions/opinions
- Show you're genuinely curious about their perspective

ENGAGEMENT STRATEGY:
- Base your question/observation on real insights from research
- Frame it through a relatable lens: "Anyone else...?" "Be honest..."
- Acknowledge different perspectives or common struggles
- Make people want to share their story

TACTICS (pick one):
1. Vulnerable Question: "What's your biggest fear about [topic from research]?"
2. Shared Experience: "We all pretend [insight], but really..."
3. Opinion Invite: "Hot take: [observation from research]. Agree or nah?"
4. Story Prompt: "Tell me about the time you first realized [concept]..."
5. Community Poll: Present real trade-offs from research, ask which matters more to THEM

STRUCTURE:
- Lead with relatable hook (question/observation that hits emotionally)
- Provide context from research in accessible language
- End with warm invitation: "Let's discuss 👇" "Your experience?" "Prove me wrong"

TONE: Curious friend genuinely wanting to hear different perspectives - humble, open, authentic

LENGTH: {length_spec}

Make people feel SAFE to share, not judged. Create space for real conversation.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "casual": {
        "name": "Casual/Observation",
        "description": "Quick thoughts, observations",
        "best_for": "Daily posting, building relatability",
        "prompt": """
YOUR MISSION: Create casual crypto content about "{topic}" for {project_name} that feels like a text from a knowledgeable friend.

RESEARCH MATERIALS:
{research_content}

CASUAL + RELATABLE STRATEGY:
- Share observation from research as if you're thinking out loud
- Use conversational language: "honestly", "real talk", "here's the thing"
- Make it feel spontaneous, not polished
- Show personality - it's okay to be opinionated or vulnerable
- Connect the insight to everyday experience

APPROACH:
- "Just realized..." "Anyone else notice..."
- "Hot take no one asked for:"
- "Can we talk about [topic] for a sec?"
- "The more I learn about [research insight], the more I..."

TONE: Smart friend casually dropping knowledge - authentic, not trying to impress

LENGTH: {length_spec}

Make it feel human, not corporate. Like you're sharing with someone you trust.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "storytelling": {
        "name": "Story/Narrative",
        "description": "Personal stories, case studies, journeys",
        "best_for": "Building connection, sharing lessons, emotional engagement",
        "prompt": """
YOUR MISSION: Tell a compelling, RELATABLE story about "{topic}" for {project_name}.

RESEARCH MATERIALS:
{research_content}

STORYTELLING WITH EMOTIONAL CONNECTION:

STORY STRUCTURE:
- Setup: Real context from research told through a specific moment/scenario
  * "Three months ago, I..." or "Picture this scenario..."
  * Make it visceral and specific
  
- Challenge: Actual problem/trade-off presented as a struggle readers relate to
  * Show the frustration, confusion, or worry
  * "Nothing worked. Every solution felt like..."
  
- Turning Point: The "aha moment" when insight from research clicked
  * "Then I learned about..."
  * Make the discovery feel earned and emotional
  
- Journey: What happened (from research) told through human experience
  * Include setbacks or surprises
  * "Here's what actually happened..."
  
- Resolution: Real outcome with emotional payoff
  * Not just "it worked" but "it felt like..."
  
- Lesson: Genuine insight that readers can apply to their journey
  * "If you're where I was..."

RELATABILITY ELEMENTS:
- Use "I/we" or "imagine you're" perspective
- Include specific details that make it real (times, feelings, doubts)
- Show vulnerability - confusion, mistakes, surprises
- Use sensory language: "felt like", "looked like", "suddenly"
- Make the lesson personal, not preachy

TONE: Friend sharing a meaningful experience over coffee - honest, reflective, generous with lessons learned

LENGTH: {length_spec}

Ground your story in real events/insights from research, but tell it in a way that makes readers feel something and see themselves in it.
""" + OUTPUT_INSTRUCTIONS
    },
    
    "relatable_thread": {
        "name": "Story-Driven Thread",
        "description": "Narrative-focused multi-tweet thread with emotional arc",
        "best_for": "Deep connection, sharing journeys, teaching through story",
        "prompt": """
YOUR MISSION: Create a STORY-DRIVEN thread (5-8 tweets) about "{topic}" for {project_name} that takes readers on an emotional journey.

CRITICAL: Ground the story in real insights from research. No invented technical claims.

RESEARCH MATERIALS:
{research_content}

NARRATIVE THREAD STRUCTURE:

Tweet 1 (Hook with Emotion):
- Start with a vulnerable moment or universal struggle
- "I used to think... until..."
- "Nobody talks about..."
- Make it intensely relatable

Tweet 2 (The Setup):
- Paint the "before" picture
- Show the pain point or confusion
- "Every day, I'd..."
- Make readers nod along

Tweet 3 (The Catalyst):
- The moment things shifted
- "Then I discovered [insight from research]..."
- Build curiosity and hope

Tweet 4-5 (The Journey):
- Show the learning process
- Use research insights as "discoveries" in the narrative
- Include surprises: "What I didn't expect was..."
- Keep it human: doubts, realizations, setbacks

Tweet 6 (The Transformation):
- How understanding [topic from research] changed things
- Be specific about the impact
- Connect technical insight to emotional outcome

Tweet 7 (The Lesson):
- What you'd tell someone starting where you were
- Wisdom earned, not just facts learned
- Based on research but framed as personal truth

Tweet 8 (The Invitation):
- Invite readers to share their journey
- "Where are you in this journey?"
- Create community around the shared experience

Each tweet:
- 200-280 characters
- Conversational, story-like flow
- Based on research but told through emotional lens
- **Bold** key moments or realizations
- Use "..." for pauses and tension

TONE: Generous storyteller sharing hard-won wisdom - vulnerable, honest, encouraging

STORY ELEMENTS:
- Specific moments, not generic statements
- Emotional words: frustrated, excited, confused, relieved, surprised
- Sensory details when possible
- Show don't tell: "My hands were shaking" not "I was nervous"
- Include self-doubt or mistakes - makes you human

Separate tweets with:
---TWEET BREAK---

Make readers FEEL the journey, not just learn the facts. Technical accuracy wrapped in human experience.
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
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            }
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100,
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

    def setup_project(self, user_id: str, project_name: str, urls: List[str], whitepaper_path: Optional[str] = None):
        """Create a new project or update existing one - CPU OPTIMIZED VERSION."""
        start_time = time.time()
        
        profile = self.get_user_profile(user_id)
        persist_dir = self._get_project_path(user_id, project_name)
        
        print(f"\n{'='*60}")
        print(f"Setting up project '{project_name}' for user: {user_id}")
        print(f"{'='*60}")
        all_documents = []
        
        url_start = time.time()
        if urls:
            print(f"\n📥 Loading {len(urls)} URLs...")
            for i, url in enumerate(urls, 1):
                try:
                    url_load_start = time.time()
                    loader = WebBaseLoader([url])
                    loader.requests_kwargs = {'timeout': 15}
                    docs = loader.load()
                    chunks = self.text_splitter.split_documents(docs)
                    all_documents.extend(chunks)
                    elapsed = time.time() - url_load_start
                    print(f"   ✓ {i}/{len(urls)}: {len(chunks)} chunks ({elapsed:.1f}s) - {url[:50]}")
                except Exception as e:
                    print(f"   ✗ {i}/{len(urls)}: Failed - {str(e)[:60]}")
            
            print(f"\n⏱️  URL Loading: {time.time() - url_start:.1f}s")
        
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
        
        vector_start = time.time()
        total_chunks = len(all_documents)
        print(f"\n🧠 Creating embeddings for {total_chunks} chunks...")
        print(f"   Estimated time: {total_chunks * 0.04:.0f}-{total_chunks * 0.08:.0f} seconds")
        
        try:
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                print("   Loading existing vector store...")
                store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                    collection_name=f"{user_id}_{project_name.lower()}"
                )
                
                batch_size = 100
                for i in range(0, len(all_documents), batch_size):
                    batch = all_documents[i:i+batch_size]
                    batch_start = time.time()
                    store.add_documents(batch)
                    batch_time = time.time() - batch_start
                    progress = min(i + batch_size, total_chunks)
                    print(f"   Progress: {progress}/{total_chunks} chunks ({batch_time:.1f}s)")
            else:
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
        
        store = self.vector_store[user_id]
        k = 8 if template_key in ["educational", "thread", "relatable_thread"] else 5
        
        docs = store.similarity_search(topic, k=k)
        
        if not docs:
            return f"Error: No relevant documents found for topic '{topic}'. Check your project setup."
        
        research_content = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG - Retrieved {len(docs)} documents")
            print(f"Total research content length: {len(research_content)} chars")
            print(f"{'='*60}")
            print("First document preview:")
            print(docs[0].page_content[:500])
            print(f"{'='*60}\n")
        
        if len(research_content) < 200:
            print("WARNING: Very little research content found.")
            print(f"   Only {len(research_content)} characters retrieved.")
            print("   Results may be generic. Consider:")
            print("   1. Adding more relevant documentation")
            print("   2. Checking if URLs loaded correctly")
            print("   3. Using a more specific topic query")
        
        length_specs = {
            "short": "400-600 words",
            "medium": "600-850 words",
            "long": "1000-1500 words"
        }
        
        prompt = template_config["prompt"].format(
            topic=topic,
            project_name=self.project_name.get(user_id, "your project"),
            research_content=research_content[:6000],
            length_spec=length_specs.get(length, length_specs["medium"])
        )
        
        if profile.brand_voice:
            prompt += f"\n\nBRAND VOICE: {profile.brand_voice}"
        
        print(f"\n🎯 Creating {template_key} content about: {topic} ({length})")
        
        system_message = SystemMessage(content="""You are a professional crypto Twitter content creator who excels at making technical content relatable and emotionally engaging.

CRITICAL RULES:
1. Use ONLY information from the research materials provided
2. Do NOT invent statistics, features, or claims
3. Connect technical accuracy with human emotion and experience
4. If you cannot find specific data, focus on explaining concepts through relatable stories/analogies
5. Output ONLY the final tweet - no explanations, no meta-commentary
6. Every data point must be traceable to the research
7. Make readers FEEL something while learning something

You create accurate, insightful content that's both credible AND emotionally resonant.""")
        
        response = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        return response.content

    def tweak_content(self, user_id: str, original_content: str, feedback: str) -> str:
        """Iterate on existing content based on feedback."""
        
        system_message = SystemMessage(content="""You are a professional crypto Twitter content creator.
You revise content based on feedback while maintaining accuracy and relatability. 
Output ONLY the revised tweet - no explanations.""")
        
        prompt = f"""
Original content:
{original_content}

User feedback:
{feedback}

Rewrite based on feedback while maintaining:
- Accuracy (don't add fake data)
- Relatability (keep it human and emotionally engaging)
- The core message

Output ONLY the revised tweet.
"""
        response = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        return response.content


def create_content_system(groq_api_key: str):
    return MultiUserContentCreator(groq_api_key)


# ═══════════════════════════════════════════════════
# USAGE EXAMPLES WITH NEW RELATABLE TEMPLATES
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    system = create_content_system(os.getenv("API_KEY"))
    
    # Register user with relatable brand voice
    system.register_user(
        "user123",
        default_template="educational",
        brand_voice="Knowledgeable friend who makes crypto accessible - technical but warm, never condescending",
        preferred_length="medium"
    )
    
    # Setup with project data
    system.setup_project(
        "user123", 
        "cysic",  
        urls=[],
        whitepaper_path=r"C:\Users\HP\Desktop\Twitter Thread Creator\Cysic whitepaper.txt"
    )
    
    # Example 1: Educational with relatability
    print("\n" + "="*60)
    print("EXAMPLE 1: RELATABLE EDUCATIONAL TWEET")
    print("="*60)
    
    tweet1 = system.create_twitter_content(
        user_id="user123",
        topic="How Cysic Network solves blockchain scalability challenges",
        template="educational",
        length="medium",
        debug=False
    )
    print(tweet1)
    
    # Example 2: Story-driven thread (NEW!)
    print("\n" + "="*60)
    print("EXAMPLE 2: STORY-DRIVEN RELATABLE THREAD")
    print("="*60)
    
    tweet2 = system.create_twitter_content(
        user_id="user123",
        topic="The journey of understanding zero-knowledge proofs through Cysic",
        template="relatable_thread",
        length="long",
        debug=False
    )
    print(tweet2)
    
    # Example 3: Engagement post with connection
    print("\n" + "="*60)
    print("EXAMPLE 3: RELATABLE ENGAGEMENT POST")
    print("="*60)
    
    tweet3 = system.create_twitter_content(
        user_id="user123",
        topic="What's your biggest frustration with current blockchain technology?",
        template="engagement",
        length="short",
        debug=False
    )
    print(tweet3)
    
    # Example 4: Show all available templates
    print("\n" + "="*60)
    print("ALL AVAILABLE TEMPLATES:")
    print("="*60)
    system.list_available_templates()
    
    # Example 5: Tweaking content based on feedback
    print("\n" + "="*60)
    print("EXAMPLE 5: ITERATING ON CONTENT")
    print("="*60)
    
    original = tweet1
    feedback = "Make it more personal and vulnerable - like I'm sharing a discovery moment"
    
    tweaked = system.tweak_content(
        user_id="user123",
        original_content=original,
        feedback=feedback
    )
    
    print("\nORIGINAL:")
    print(original[:200] + "...")
    print("\nTWEAKED BASED ON FEEDBACK:")
    print(tweaked)