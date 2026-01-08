#!/usr/bin/env python3
"""
================================================================================
STANDALONE MATCHING SCORE TEST SCRIPT
================================================================================

This script is an ISOLATED COPY of the actual matching score solution, 
using the SAME PROMPTS from the main project (core/prompts.py).

PURPOSE:
--------
Test and validate the criteria-based matching score approach using ACTUAL LLM API calls
with the same prompts as the production system.

WHAT IT DOES:
-------------
1. Extracts rubric criteria from a job posting (REAL LLM API call to Claude)
2. Scores a candidate against each criterion (REAL LLM API call to Claude)
3. Calculates weighted final matching score (actual calculation logic)

INPUT FORMAT:
-------------
- job_posting: String containing the job description
- cv_profile: String containing the candidate's CV/resume

OUTPUT FORMAT:
--------------
- Rubric: List of criteria with weights and descriptions
- Criteria Scores: Score (0-100) for each criterion with evidence and gaps
- Final Score: Weighted average of all criteria scores

REQUIREMENTS:
-------------
- Python 3.10+
- requests package: pip install requests
- OPENROUTER_API_KEY environment variable set
- (Optional) Langfuse: pip install langfuse python-dotenv

USAGE:
------
Setup:
    export OPENROUTER_API_KEY='your-openrouter-api-key-here'
    pip install requests langfuse python-dotenv

Basic (run all scenarios):
    python test_matching_score.py

================================================================================
LANGFUSE INTEGRATION - STEP-BY-STEP GUIDE
================================================================================

WHAT IS LANGFUSE?
-----------------
Langfuse is an observability platform for LLM applications. It tracks:
- All LLM API calls (prompts, outputs, tokens, costs, latency)
- Quality metrics and scores
- Prompt versions and A/B testing
- Performance analytics and debugging

Think of it as "Google Analytics for your AI application"

STEP 1: GET LANGFUSE CREDENTIALS
---------------------------------
Option A: Langfuse Cloud (Easiest - Free tier available)
  1. Go to https://cloud.langfuse.com
  2. Sign up for free account
  3. Create a project
  4. Go to Settings → API Keys
  5. Copy: Public Key, Secret Key, Host URL

Option B: Self-hosted (Advanced)
  1. Follow: https://langfuse.com/docs/deployment/self-host
  2. Deploy using Docker
  3. Get your credentials from your instance

STEP 2: CONFIGURE ENVIRONMENT VARIABLES
----------------------------------------
Add these to tests/test_matching/.env file:

    OPENROUTER_API_KEY=sk-or-v1-your-key-here
    LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
    LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
    LANGFUSE_HOST=https://cloud.langfuse.com

The script will automatically load these credentials.

STEP 3: RUN THE SCRIPT
-----------------------
Just run the script normally:

    python test_matching_score.py

If Langfuse is configured, you'll see: "✓ Langfuse observability enabled"
If not configured: "⚠ Langfuse not configured"

STEP 4: VIEW RESULTS IN LANGFUSE DASHBOARD
-------------------------------------------
1. Go to https://cloud.langfuse.com (or your self-hosted URL)
2. Navigate to your project
3. Click "Traces" in the sidebar

You'll see all your matching score calculations with:
- Complete execution flow (rubric extraction → scoring → final score)
- Token usage and costs
- Latency for each operation
- Full prompts and outputs
- Quality scores

UNDERSTANDING LANGFUSE CONCEPTS
--------------------------------

1. TRACE (Top Level)
   - Represents ONE complete matching score calculation
   - Example: "Matching Score - Alice vs Senior Developer"
   - Contains: metadata, user_id, tags, scores
   - Benefits: Track end-to-end performance, filter by user/tag

2. SPAN (Mid Level - Operations)
   - Represents a sub-operation within a trace
   - Examples: "Rubric Extraction", "Criteria Scoring"
   - Contains: input, output, latency, status
   - Benefits: See which part is slow, track success/failure

3. GENERATION (Low Level - LLM Calls)
   - Represents ONE LLM API call
   - Contains: prompt, response, model, tokens, cost
   - Benefits: Track exact prompts/outputs, compare models

4. SCORES (Quality Metrics)
   - Numeric values attached to traces
   - Examples: matching_score (0-100), quality (0-1)
   - Benefits: Filter high/low scores, track improvements

Hierarchy Example:
    Trace: "matching_score" (Alice)
      ├─ Span: "rubric_extraction"
      │   └─ Generation: "rubric_extraction_llm" (Claude call)
      └─ Span: "criteria_scoring"
          └─ Generation: "criteria_scoring_llm" (Claude call)
      Scores: matching_score=85, quality=1

LANGFUSE DASHBOARD FEATURES
----------------------------

1. TRACES VIEW
   - See all matching score calculations
   - Filter by: date, user, tag, score range
   - Sort by: latency, cost, score
   - Click any trace to see details

2. GENERATIONS VIEW
   - See all LLM API calls
   - Filter by: model, latency, token count
   - Compare prompt performance
   - Identify expensive calls

3. ANALYTICS
   - Token usage over time
   - Cost tracking
   - Latency distribution
   - Success/error rates

4. PROMPTS (Advanced)
   - Store prompts in Langfuse UI
   - Version control for prompts
   - A/B test different versions
   - Roll back to previous versions

USING LANGFUSE FOR PROMPT IMPROVEMENT
--------------------------------------

1. Baseline: Run tests and track scores in Langfuse
2. Improve: Modify CRITERIA_SCORING_PROMPT
3. Test: Run same tests again
4. Compare: View both versions in Langfuse dashboard
5. Decide: Keep the version with better scores/performance

Example workflow:
   - Run test with current prompt → avg score: 65
   - Modify prompt to be more objective → avg score: 72
   - Compare in Langfuse → see improvement
   - Keep the better prompt

ADVANCED: PROMPT MANAGEMENT IN LANGFUSE UI
-------------------------------------------

1. In Langfuse dashboard, go to "Prompts"
2. Create prompt: "criteria_scoring"
3. Paste CRITERIA_SCORING_PROMPT content
4. Publish version 1
5. In code, fetch prompt from Langfuse:
   
   prompt = langfuse.get_prompt("criteria_scoring")
   messages = prompt.compile(rubric=rubric_text, cv=cv_profile)

6. To test new version:
   - Edit prompt in Langfuse UI
   - Publish as version 2
   - Compare v1 vs v2 performance
   - Promote better version to production

TROUBLESHOOTING
---------------

Q: "⚠ Langfuse not configured"
A: Check .env file has LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY

Q: "⚠ Langfuse not installed"
A: Run: pip install langfuse python-dotenv

Q: No traces appear in dashboard
A: Check credentials, ensure LANGFUSE_HOST is correct

Q: Want to disable Langfuse temporarily
A: Remove or comment out LANGFUSE_PUBLIC_KEY in .env

Q: Too many traces cluttering dashboard
A: Use tags to filter (e.g., tag="production" vs tag="test")

RESOURCES
---------
- Docs: https://langfuse.com/docs
- Python SDK: https://langfuse.com/docs/sdk/python
- Prompt Management: https://langfuse.com/docs/prompts
- Scoring: https://langfuse.com/docs/scores

================================================================================

Single Test:
    job_posting = "Your job description here..."
    cv_profile = "Candidate's CV text here..."
    test_scenario("Your Scenario Name", job_posting, cv_profile)

Multiple Profiles (uses cache for consistency):
    job_posting = "Senior Developer position..."
    candidates = [
        ("Alice", "Alice's CV text..."),
        ("Bob", "Bob's CV text..."),
        ("Carol", "Carol's CV text...")
    ]
    results = test_multiple_profiles(job_posting, candidates)
    
    # This will:
    # 1. Generate rubric once (cached)
    # 2. Score all 3 candidates against SAME rubric
    # 3. Show comparison summary

HOW IT WORKS:
-------------
1. Uses REAL LLM API calls via OpenRouter (default: anthropic/claude-3.5-sonnet)
2. Prompts loaded from prompts.py (same as production)
3. Rubric extraction: LLM extracts 6-10 weighted criteria from job posting
4. **CACHING**: Rubric is cached based on job posting hash (ensures consistency)
5. Criteria scoring: LLM scores candidate 0-100 on each criterion
6. Final score: Weighted average calculation (same logic as production)

RUBRIC CACHING:
---------------
**Why?** When testing multiple candidates for the same job, you want consistent 
evaluation criteria. The cache ensures the exact same rubric is used for all candidates.

**How?**
- Rubric is cached in `.rubric_cache/` directory
- Cache key = SHA256 hash of job posting (first 16 chars)
- Same job posting = same rubric (even across different script runs)

**Cache Management:**
    # List cached rubrics
    list_cached_rubrics()
    
    # Clear all cached rubrics
    clear_rubric_cache()
    
    # Disable caching (set at top of script)
    ENABLE_CACHE = False

CHANGING THE MODEL:
-------------------
Edit the script and change OPENROUTER_MODEL variable:
- "anthropic/claude-3.5-sonnet" (default)
- "openai/gpt-4-turbo"
- "google/gemini-pro-1.5"
- Or any other model available on OpenRouter

**Note:** Changing the model invalidates the cache (different model = different rubric)

VALIDATION:
-----------
This script helps validate that:
- The scoring logic produces expected ranges for different scenarios
- The weighted average calculation works correctly
- The rubric extraction identifies appropriate criteria
- The evidence and gap reporting is clear

EXPECTED SCORE RANGES:
----------------------
- Perfect Match: 90-100
- Strong Fit: 80-89
- Good Fit: 65-79
- Moderate Fit: 50-64
- Weak Fit: 30-49
- Poor Fit: 0-29

AUTHOR: Abdelaziz Bellout
DATE: 2025-11-24
================================================================================
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
import sys
import os
import re
import requests
import hashlib
import pickle
from pathlib import Path
from prompts import (
    RUBRIC_EXTRACTION_PROMPT, 
    CRITERIA_SCORING_PROMPT_WITH_REASONING as CRITERIA_SCORING_PROMPT, 
    QUALIFICATION_GENERATION_PROMPT,
    QUALIFICATION_SUMMARY
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this script
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# ============================================================================
# LANGFUSE INTEGRATION
# ============================================================================
# Langfuse is an observability platform that tracks:
# 1. All LLM calls (prompts, inputs, outputs)
# 2. Costs and token usage
# 3. Latency and performance
# 4. Quality scores and feedback
# 5. Prompt versions and A/B testing

try:
    from langfuse import Langfuse
    
    # Initialize Langfuse client
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    
    if LANGFUSE_ENABLED:
        print("✓ Langfuse observability enabled (manual tracing)")
    else:
        print("⚠ Langfuse not configured (set LANGFUSE_PUBLIC_KEY in .env)")
except ImportError as e:
    print(f"⚠ Langfuse not installed: {e}")
    print("  Run: pip install langfuse")
    langfuse = None
    LANGFUSE_ENABLED = False
except Exception as e:
    print(f"⚠ Langfuse initialization error: {e}")
    langfuse = None
    LANGFUSE_ENABLED = False


# ============================================================================
# FIT LEVEL SCALE (for consistent qualification assessment)
# ============================================================================

FIT_LEVEL_SCALE = [
    (90, "Exceptional Fit"),
    (80, "Strong Fit"),
    (65, "Good Fit"),
    (50, "Moderate Fit"),
    (30, "Weak Fit"),
    (0, "Poor Fit")
]


def get_fit_level(matching_score: int) -> str:
    """
    Determine fit level from matching score using the standard scale.
    This ensures consistency across qualification note, summary, and matching score.
    
    Args:
        matching_score: Integer score from 0-100
        
    Returns:
        Fit level string (e.g., "Strong Fit", "Good Fit")
    """
    for min_score, fit_level in FIT_LEVEL_SCALE:
        if matching_score >= min_score:
            return fit_level
    return "Poor Fit"


# ============================================================================
# DATA MODELS (mirroring actual project models)
# ============================================================================

@dataclass
class RubricCriterion:
    """A single evaluation criterion with weight."""
    name: str
    weight: float
    description: str
    is_required: bool


@dataclass
class EvaluationRubric:
    """Complete rubric with all criteria."""
    criteria: List[RubricCriterion]
    total_weight: float


@dataclass
class CriterionScore:
    """Score for a single criterion."""
    criteria_name: str
    score: float
    reasoning: str # Added for stability (Chain of Thought)
    evidence: str
    gap: str


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get API key from environment variable or .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Removed for OpenRouter-only support
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") # Removed for OpenRouter-only support
# FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") # Removed for OpenRouter-only support
# CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY") # Removed for OpenRouter-only support
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found!")
    print("Please set it in one of these ways:")
    print("  1. Create a .env file with: OPENROUTER_API_KEY=your-key-here")
    print("  2. Set environment variable: export OPENROUTER_API_KEY='your-key-here'")
    print("  3. Install python-dotenv: pip install python-dotenv")
    sys.exit(1)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
# TOGETHER_BASE_URL = "https://api.together.xyz/v1/chat/completions" # Removed for OpenRouter-only support
# FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
# CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# Available Models
CLAUDE_HAIKU_OPENROUTER = "anthropic/claude-haiku-4.5"
GEMINI_FLASH_OPENROUTER = "google/gemini-2.5-flash-preview-09-2025"
GEMINI_FLASH_LITE_OPENROUTER = "google/gemini-2.5-flash-lite"
GPT_OSS_120B_OPENROUTER = "openai/gpt-oss-120b:exacto"
MISTRAL_14B_2512_OPENROUTER = "mistralai/ministral-14b-2512"
GROK_4_FAST_OPENROUTER = "x-ai/grok-4-fast"
GEMINI_3_FLASH_OPENROUTER = "google/gemini-3-flash-preview"
GROQ_KIMI_K2="groq::moonshotai/kimi-k2-instruct"
GROQ_LLAMA_4_MAVERICK_INSTRUCT="groq::meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_LLAMA_4_SCOUT_INSTRUCT = "groq::meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_QWEN_3_32B = "groq::qwen/qwen3-32b"
GROG_GPT_OSS_120=  "groq::openai/gpt-oss-120b"  

# Direct Groq Models
GROQ_LLAMA_3_3_70B = "groq::meta-llama/llama-3.3-70b-instruct"
# GROQ_LLAMA_3_1_8B = "groq:meta-llama/llama-3.1-8b-instruct"
# GROQ_MIXTRAL_8X7B = "groq:meta-llama/mixtral-8x7b-instruct"
# GROQ_LLAMA_4 = "groq:meta-llama/llama-4-405b" # Assuming this exists or is wanted
TOGETHER_GLM_4_5_AIR_FP8 = "together::zai-org/GLM-4.5-Air-FP8"
FIREWORKS_GLM_4_7 = "fireworks::zai-org/glm-4.7"
FIREWORKS_MINIMAX_M2P1 = "fireworks::minimax/minimax-m2p1"
CEREBRAS_GLM_4_6 = "cerebras::z-ai/glm-4.6"
CEREBRAS_GLM_4_7 = "cerebras::z-ai/glm-4.7"
CEREBRAS_QWEN_3_235B = "cerebras::qwen/qwen-3-235b-instruct"
CEREBRAS_LLAMA_3_3_70B = "cerebras::meta-llama/llama-3.3-70b-instruct"

# Direct Cerebras Models
DIRECT_CEREBRAS_GLM_4_6 = "direct-cerebras::zai-glm-4.6"
DIRECT_CEREBRAS_LLAMA_3_3_70B = "direct-cerebras::llama-3.3-70b"

# Default model (can be overridden)
OPENROUTER_MODEL = CLAUDE_HAIKU_OPENROUTER

# Model display names for UI
MODEL_NAMES = {
    CLAUDE_HAIKU_OPENROUTER: "Claude Haiku 4.5",
    GEMINI_FLASH_OPENROUTER: "Gemini 2.5 Flash Preview",
    GEMINI_FLASH_LITE_OPENROUTER: "Gemini 2.5 Flash Lite",
    GEMINI_3_FLASH_OPENROUTER: "Gemini 3 Flash Preview",
    MISTRAL_14B_2512_OPENROUTER: "Mistral 14B 2512",
    GROK_4_FAST_OPENROUTER: "Grok 4.1 Fast",
    GPT_OSS_120B_OPENROUTER: "GPT OSS 120B (Exacto)",
    GROQ_KIMI_K2: "Kimi K2 Instruct (OpenRouter - Groq)",
    GROQ_LLAMA_4_MAVERICK_INSTRUCT: "Llama 4 Maverick (OpenRouter - Groq)",
    GROQ_LLAMA_4_SCOUT_INSTRUCT: "Llama 4 Scout (OpenRouter - Groq)",
    # GROQ_LLAMA_4: "Llama 4 405B (OpenRouter - Groq)",
    GROG_GPT_OSS_120: "GPT OSS 120B (OpenRouter - Groq)",
    GROQ_QWEN_3_32B: "Qwen 3 32B (OpenRouter - Groq)",
    GROQ_LLAMA_3_3_70B: "Llama 3.3 70B (OpenRouter - Groq)",
    # GROQ_LLAMA_3_1_8B: "Llama 3.1 8B (OpenRouter - Groq)",
    # GROQ_MIXTRAL_8X7B: "Mixtral 8x7b (OpenRouter - Groq)",
    TOGETHER_GLM_4_5_AIR_FP8: "GLM-4.5 Air FP8 (OpenRouter - Together)",
    FIREWORKS_GLM_4_7: "GLM 4.7 (OpenRouter - Fireworks)",
    FIREWORKS_MINIMAX_M2P1: "Minimax M2P1 (OpenRouter - Fireworks)",
    CEREBRAS_GLM_4_6: "GLM 4.6 (OpenRouter - Cerebras)",
    CEREBRAS_GLM_4_7: "GLM 4.7 (OpenRouter - Cerebras)",
    CEREBRAS_QWEN_3_235B: "Qwen 3 235B (OpenRouter - Cerebras)",
    CEREBRAS_LLAMA_3_3_70B: "Llama 3.3 70B (OpenRouter - Cerebras)",
    DIRECT_CEREBRAS_GLM_4_6: "GLM 4.6 (Direct Cerebras)",
    DIRECT_CEREBRAS_LLAMA_3_3_70B: "Llama 3.3 70B (Direct Cerebras)",
}

# Cache configuration
CACHE_DIR = Path(__file__).parent / ".rubric_cache"
CACHE_DIR.mkdir(exist_ok=True)
ENABLE_CACHE = False  # Set to False to disable caching

# ============================================================================
# PROMPTS (copied from actual project)
# ============================================================================

# CRITERIA_SCORING_PROMPT is imported from prompts.py at the top of the file

# Rubric extraction prompt (actual prompt used in project)
RUBRIC_EXTRACTION_PROMPT = """You are an AI that extracts evaluation criteria from job postings.

Given a job posting, extract 6-10 weighted criteria that should be used to evaluate candidates.

**CRITICAL INSTRUCTIONS FOR TECHNICAL SKILLS:**
1. **Always include required level/seniority in the description**
   - Example: "Strong React.js expertise (Senior level, 5+ years)"
   - Example: "Python proficiency (Medior level, 3+ years)"
   - Example: "Basic SQL knowledge (Junior level, 1+ years)"

2. **Weight criteria based on importance:**
   - Critical must-haves: 15-25%
   - Important requirements: 10-15%
   - Nice-to-haves: 5-10%
   - Total must equal 100%

3. **Categories to consider:**
   - Job Title Match
   - Experience Years
   - Technical Skills (break down by technology, always include required level)
   - Soft Skills
   - Languages
   - Industry Experience
   - Education

**CRITICAL OUTPUT INSTRUCTIONS:**
1. Return ONLY a valid JSON object.
2. DO NOT include introductory or concluding text.
3. DO NOT include markdown formatting outside the JSON block.
4. The response must be directly parseable by `json.loads()`.

Return ONLY a valid JSON object with this structure:
{
  "criteria": [
    {
      "name": "Criterion Name",
      "weight": 15.0,
      "description": "Detailed description including required level",
      "is_required": true
    }
  ],
  "total_weight": 100.0
}
"""



# ============================================================================
# CORE FUNCTIONS (mirroring actual project logic)
# ============================================================================

def calculate_matching_score(rubric: EvaluationRubric, criteria_scores: List[CriterionScore]) -> dict:
    """
    Calculate weighted matching score based on rubric and criteria scores.
    
    Args:
        rubric: The evaluation rubric with criteria and weights
        criteria_scores: List of scores for each criterion
        
    Returns:
        dict with final_score, breakdown, and details
    """
    # Create weight map
    weight_map = {c.name: c.weight for c in rubric.criteria}
    
    # Calculate weighted average
    total_weight = 0
    weighted_sum = 0
    breakdown = []
    
    # Debug: Print available criterion names
    print(f"DEBUG - Available rubric criteria names: {list(weight_map.keys())}")
    print(f"DEBUG - Criteria scores received: {[cs.criteria_name for cs in criteria_scores]}")
    
    for criterion_score in criteria_scores:
        weight = weight_map.get(criterion_score.criteria_name, 0)
        
        if weight == 0:
            # Try to find a match (fuzzy matching)
            print(f"WARNING: Criterion '{criterion_score.criteria_name}' not found in rubric!")
            print(f"  Available criteria: {list(weight_map.keys())}")
            # Try to find partial match
            for rubric_name in weight_map.keys():
                if rubric_name.lower() in criterion_score.criteria_name.lower() or criterion_score.criteria_name.lower() in rubric_name.lower():
                    print(f"  Possible match found: '{rubric_name}'")
                    weight = weight_map[rubric_name]
                    criterion_score.criteria_name = rubric_name  # Update to correct name
                    break
        
        if weight > 0:
            contribution = criterion_score.score * (weight / 100)
            weighted_sum += contribution
            total_weight += weight
            
            # Ensure evidence, gap, and reasoning are strings (not None)
            evidence = criterion_score.evidence if criterion_score.evidence else ""
            gap = criterion_score.gap if criterion_score.gap else ""
            reasoning = getattr(criterion_score, 'reasoning', "") or ""
            
            breakdown.append({
                "criterion": criterion_score.criteria_name,
                "score": criterion_score.score,
                "weight": weight,
                "contribution": round(contribution, 2),
                "reasoning": reasoning,
                "evidence": evidence,
                "gap": gap
            })
            
            # Debug: Log if evidence/gap are empty
            if not evidence and not gap:
                print(f"WARNING: Empty evidence/gap for criterion: {criterion_score.criteria_name}")
        else:
            print(f"ERROR: Could not match criterion '{criterion_score.criteria_name}' - skipping from breakdown")
    
    # Calculate final score
    if total_weight > 0:
        final_score = round(weighted_sum)
    else:
        # Fallback to simple average
        if len(criteria_scores) > 0:
            final_score = round(sum(c.score for c in criteria_scores) / len(criteria_scores))
        else:
            final_score = 0
            print("WARNING: No criteria scores to calculate average. Defaulting to 0.")
    
    return {
        "final_score": final_score,
        "total_weight_used": total_weight,
        "breakdown": breakdown
    }


def clean_llm_json_response(response_text: str) -> str:
    """
    Clean LLM response by removing reasoning blocks (<think>...</think>),
    removing pagination artifacts (e.g., "Page 1 of 2"),
    and extracting content within JSON markdown blocks if present.
    Also attempts to repair truncated JSON if possible.
    Includes heuristics to quote unquoted description fields and insert missing
    commas before following keys (common LLM formatting slips).
    """
    import re
    
    # 1. Remove <think> blocks (used by reasoning models like Qwen 3 or DeepSeek R1)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    
    # If <think> still exists (e.g. unclosed tag), remove everything until the first '{'
    if '<think>' in response_text:
        think_idx = response_text.find('<think>')
        json_start = response_text.find('{', think_idx)
        if json_start != -1:
            response_text = response_text[json_start:]
        else:
            response_text = response_text[think_idx + 7:]

    # 2. Remove pagination artifacts like "Page 1 of 2" or "Page 1 / 2" or standalone "Page 10"
    # These often appear in the middle of JSON from certain providers/models
    response_text = re.sub(r'Page \d+ (?:of|/) \d+', '', response_text, flags=re.IGNORECASE)
    response_text = re.sub(r'Page\s+\d+\s*', '', response_text, flags=re.IGNORECASE)
    # Remove stray "Page {" or ", Page {" tokens that break JSON objects
    response_text = re.sub(r',?\s*Page\s*\{', '{', response_text, flags=re.IGNORECASE)
    # Remove/normalize "PageRoute" artifacts that appear before weight values
    response_text = re.sub(r'"weight\s+PageRoute:\s*",? ?', '"weight": ', response_text, flags=re.IGNORECASE)
    response_text = re.sub(r'PageRoute\s*:\s*', '', response_text, flags=re.IGNORECASE)
    
    response_text = response_text.strip()
    
    # 3. Extract content from markdown JSON blocks
    json_blocks = re.findall(r'```json\s*(.*?)\s*(?:```|$)', response_text, flags=re.DOTALL)
    if json_blocks:
        response_text = json_blocks[-1].strip()
    else:
        code_blocks = re.findall(r'```\s*(.*?)\s*(?:```|$)', response_text, flags=re.DOTALL)
        if code_blocks:
            response_text = code_blocks[-1].strip()
            
    # 4. If no markdown blocks, try to find the first '{' and last '}'
    if not (response_text.startswith('{') and response_text.endswith('}')):
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1:
            if end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
            else:
                # If no closing brace found, take everything from start brace
                # and let the repair logic (Step 5) handle it
                response_text = response_text[start_idx:]
        else:
            # No JSON object found; return empty JSON to avoid crashes
            return "{}"
            
    response_text = response_text.strip()
    
    # 5. Attempt to repair truncated JSON
    if response_text.startswith('{') and not response_text.endswith('}'):
        print("⚠ Detected truncated JSON, attempting repair...")
        
        # Remove partial keys/values at the very end (e.g., , "descrip )
        # Find the last occurrence of either a complete object } or a complete value
        last_comma = response_text.rfind(',')
        last_brace = response_text.rfind('}')
        last_bracket = response_text.rfind(']')
        
        # Cut back to the last structural comma or brace
        cut_point = max(last_brace, last_bracket)
        
        if cut_point != -1:
            response_text = response_text[:cut_point + 1]
            
            # Close opened structures
            open_braces = response_text.count('{') - response_text.count('}')
            open_brackets = response_text.count('[') - response_text.count(']')
            
            response_text += ']' * open_brackets
            response_text += '}' * open_braces
        else:
            # Last resort: just try to close the main object
            if not response_text.endswith('}'):
                response_text += '"}' if response_text.count('"') % 2 != 0 else '}'

    # 6b. Balance any remaining unmatched brackets/braces (helps when array/object not closed)
    open_brackets = response_text.count('[') - response_text.count(']')
    open_braces = response_text.count('{') - response_text.count('}')
    if open_brackets > 0:
        response_text += ']' * open_brackets
    if open_braces > 0:
        response_text += '}' * open_braces

    # 6. Quote unquoted description values and add missing comma before next key
    # Pattern: "description": Some text "is_required": true  --> quote text + add comma
    def _quote_desc(match):
        prefix = match.group(1)
        val = match.group(2).strip()
        return f'{prefix}"{val}"'

    # Quote when description value is unquoted and immediately followed by "is_required"
    response_text = re.sub(
        r'("description"\s*:\s*)(?!")([^\n\r\{\}\[\],]*?)(?="\s*is_required")',
        _quote_desc,
        response_text,
        flags=re.IGNORECASE
    )
    # Ensure comma exists between description and next "is_required"
    response_text = re.sub(
        r'("description"\s*:\s*"[^"]*")\s*("is_required")',
        r'\1, \2',
        response_text,
        flags=re.IGNORECASE
    )

    # 7. Fix missing "name" keys formatted as { "Some Name", "weight": ... }
    response_text = re.sub(
        r'\{\s*"([^"]+)"\s*(?!:)\s*,\s*"weight"',
        lambda m: f'{{ "name": "{m.group(1).strip()}", "weight"',
        response_text
    )

    return response_text.strip()


def call_openrouter(
    messages: List[Dict[str, str]], 
    max_tokens: int = 2000,
    generation_name: str = "openrouter_call",
    langfuse_parent=None,
    langfuse_prompt=None,
    session_id: str = None,
    model: str = None
) -> tuple[str, float]:
    """
    Make an API call to OpenRouter or Groq with Langfuse observability.
    Includes retry logic for rate limits (429).
    """
    import time
    import random
    
    # Use the provided model or fall back to default
    selected_model = model if model else OPENROUTER_MODEL
    
    # Strip internal provider prefix if present (e.g. "groq::meta-llama/..." -> "meta-llama/...")
    actual_model_id = selected_model
    if "::" in selected_model:
        actual_model_id = selected_model.split("::")[-1]
    
    # Determine if this is a direct Groq or Together call
    is_direct_groq = selected_model.startswith("groq::") or selected_model in [
        GROQ_KIMI_K2, 
        GROQ_LLAMA_4_MAVERICK_INSTRUCT, 
        GROQ_LLAMA_4_SCOUT_INSTRUCT,
        # GROQ_LLAMA_4,
        GROG_GPT_OSS_120,
        GROQ_QWEN_3_32B,
        GROQ_LLAMA_3_3_70B, 
        # GROQ_LLAMA_3_1_8B, 
        # GROQ_MIXTRAL_8X7B
    ]
    is_together = selected_model.startswith("together::") or selected_model in [TOGETHER_GLM_4_5_AIR_FP8]
    is_fireworks = selected_model.startswith("fireworks::") or selected_model in [FIREWORKS_GLM_4_7, FIREWORKS_MINIMAX_M2P1]
    is_cerebras = selected_model.startswith("cerebras::") or selected_model in [CEREBRAS_GLM_4_7, CEREBRAS_GLM_4_6, CEREBRAS_QWEN_3_235B, CEREBRAS_LLAMA_3_3_70B]
    is_gemini = selected_model in [GEMINI_3_FLASH_OPENROUTER, GEMINI_FLASH_OPENROUTER, GEMINI_FLASH_LITE_OPENROUTER]
    is_direct_cerebras = selected_model.startswith("direct-cerebras::") or selected_model in [DIRECT_CEREBRAS_GLM_4_6, DIRECT_CEREBRAS_LLAMA_3_3_70B]
    
    api_url = OPENROUTER_BASE_URL
    api_key = OPENROUTER_API_KEY
    
    if is_direct_cerebras:
        api_url = CEREBRAS_BASE_URL
        api_key = CEREBRAS_API_KEY
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via Direct Cerebras API] {generation_name}...")
    elif is_direct_groq:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via OpenRouter (Provider: Groq)] {generation_name}...")
    elif is_together:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via OpenRouter (Provider: Together)] {generation_name}...")
    elif is_fireworks:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via OpenRouter (Provider: Fireworks)] {generation_name}...")
    elif is_cerebras:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via OpenRouter (Provider: Cerebras)] {generation_name}...")
    elif is_gemini:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables!")
        print(f"\n[LLM CALL via OpenRouter (Prioritizing Google AI Studio)] {generation_name}...")
    else:
        api_url = OPENROUTER_BASE_URL
        api_key = OPENROUTER_API_KEY
        print(f"\n[LLM CALL via OpenRouter] {generation_name}...")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    headers["HTTP-Referer"] = "https://github.com/wiggli-parser"
    headers["X-Title"] = "Wiggli Parser Test"
    
    print(f"Model: {selected_model}")
    
    # If Together AI, bump max_tokens to reduce truncation risk
    if is_together and max_tokens < 6000:
        max_tokens = 6000
    elif is_fireworks and max_tokens > 4096:
        max_tokens = 4096
    elif is_cerebras and max_tokens < 6000:
        max_tokens = 6000

    data = {
        "model": actual_model_id,
        "messages": messages,
        # --- CONSISTENCY PARAMETERS ---
        "temperature": 0,      # Removes randomness
        "top_p": 1,           # Restricts sampling to top probability
        "seed": 42            # Forces deterministic output (if supported by model)
    }

    # If this is a Groq, Together, Fireworks or Cerebras model being called via OpenRouter, specify the provider
    if is_direct_groq:
        data["provider"] = {
            "order": ["Groq", "DeepInfra", "Novita"],
            "allow_fallbacks": True
        }
    elif is_together:
        data["provider"] = {
            "order": ["Together", "DeepInfra"],
            "allow_fallbacks": True
        }
    elif is_fireworks:
        data["provider"] = {
            "order": ["Fireworks", "DeepInfra"],
            "allow_fallbacks": True
        }
    elif is_cerebras:
        data["provider"] = {
            "order": ["Cerebras"],
            "allow_fallbacks": True
        }
    elif is_gemini:
        data["provider"] = {
            "order": ["Vertex AI", "Google AI Studio"],
            "sort": "latency"
        }
    
    if is_cerebras or is_direct_cerebras:
        data["max_completion_tokens"] = max_tokens
    else:
        data["max_tokens"] = max_tokens
    
    # Enable JSON mode ONLY for rubric extraction and criteria scoring
    if is_fireworks or is_together or is_direct_groq or is_cerebras or is_direct_cerebras:
        if generation_name in ["rubric_extraction_llm", "criteria_scoring_llm"]:
            data["response_format"] = {"type": "json_object"}
    
    # LANGFUSE: Create generation manually (v3.x API with session grouping)
    generation = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Build generation parameters
            gen_params = {
                "name": generation_name,
                "model": selected_model,
                "model_parameters": {
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42
                },
                "input": messages,
                "prompt": langfuse_prompt
            }
            
            # Create generation using the client directly
            # Note: We bypass propagate_attributes here because it is not thread-safe
            # when used with ThreadPoolExecutor in the compare_all_models script.
            if langfuse_parent:
                if hasattr(langfuse_parent, 'generation'):
                    generation = langfuse_parent.generation(**gen_params)
                elif hasattr(langfuse_parent, 'start_generation'):
                    generation = langfuse_parent.start_generation(**gen_params)
            else:
                if hasattr(langfuse, 'generation'):
                    generation = langfuse.generation(**gen_params)
                elif hasattr(langfuse, 'start_generation'):
                    generation = langfuse.start_generation(**gen_params)
            
            # Add session_id after creation if provided
            if generation and session_id:
                try:
                    generation.update(session_id=session_id)
                except Exception as e:
                    print(f"⚠ Could not set session_id on generation: {e}")
                    
        except Exception as e:
            print(f"⚠ Langfuse generation creation failed: {e}")
            generation = None

    # Implement retry logic for 429
    max_retries = 10
    base_delay = 5 # seconds
    provider_name = "Fireworks" if is_fireworks else "Together" if is_together else "Groq" if is_direct_groq else "Cerebras" if (is_cerebras or is_direct_cerebras) else "OpenRouter"
    
    # Track OpenRouter actual provider if possible
    actual_provider = provider_name if provider_name != "OpenRouter" else "OpenRouter"

    for attempt in range(max_retries):
        # Track actual LLM API call time (excluding Langfuse overhead)
        llm_start_time = time.time()
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            llm_duration = time.time() - llm_start_time
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ {provider_name} Rate limited (429). Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached for 429 error.")
            
            # Check for other HTTP errors
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = json.dumps(error_json, indent=2)
                except:
                    pass
                # Update generation with error
                if generation:
                    generation.update(status_message=error_detail, level="ERROR")
                    generation.end()
                raise ValueError(f"{provider_name} API error (status {response.status_code}): {error_detail}")
            
            response.raise_for_status()
            
            # Parse response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                if generation:
                    generation.update(status_message=str(e), level="ERROR")
                    generation.end()
                raise ValueError(f"Invalid JSON response from API: {response.text[:500]}") from e
            
            # Check for API-level errors in response
            if "error" in result:
                error_msg = result.get("error") or {}
                if isinstance(error_msg, dict):
                    error_detail = error_msg.get("message", str(error_msg))
                else:
                    error_detail = str(error_msg)
                
                # Check for 429 within the JSON response (some providers do this)
                if "429" in error_detail or "rate limit" in error_detail.lower() or "quota" in error_detail.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ {provider_name} Rate limited (API Error). Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue

                if generation:
                    generation.update(status_message=error_detail, level="ERROR")
                    generation.end()
                raise ValueError(f"{provider_name} API error: {error_detail}")
            
            # Extract content
            choices = result.get("choices")
            if not choices:
                error_msg = f"No choices in API response. Full response: {json.dumps(result, indent=2)[:500]}"
                if generation:
                    generation.update(status_message=error_msg, level="ERROR")
                    generation.end()
                raise ValueError(error_msg)
                
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content")
            
            # If content is empty but reasoning is present (some models like GLM-4.7 do this)
            if not content and "reasoning" in message:
                content = message["reasoning"]
            
            # If still no content, check if it's in the choice itself (some older formats)
            if not content:
                content = choice.get("text")
                
            finish_reason = choice.get("finish_reason")
            
            if finish_reason == "length":
                print(f"⚠ WARNING: LLM response was TRUNCATED (finish_reason: length). Increase max_tokens!")
            
            # Check if content is empty
            if not content or not content.strip():
                error_msg = f"Empty response from API. Full response: {json.dumps(result, indent=2)[:500]}"
                if generation:
                    generation.update(status_message=error_msg, level="ERROR")
                    generation.end()
                raise ValueError(error_msg)
            
            # LANGFUSE: Update generation with output
            if generation:
                try:
                    # Extract token usage
                    usage = result.get("usage") or {}
                    # Update output and usage first
                    generation.update(
                        output=content,
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    )
                    # Add session_id if provided (manual tracing is thread-safe)
                    if session_id:
                        try:
                            generation.update(session_id=session_id)
                        except:
                            pass
                    # Then end the generation
                    generation.end()
                    
                    if session_id:
                        print(f"✓ Generation completed with session_id: {session_id}")
                except Exception as e:
                    print(f"⚠ Langfuse generation update failed: {e}")
            
            # Return content and actual LLM call duration (excluding Langfuse overhead)
            print(f"⏱️  LLM API call took: {llm_duration:.2f}s")
            return content, llm_duration

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Network error: {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                if generation:
                    generation.update(status_message=str(e), level="ERROR")
                    generation.end()
                raise


def get_job_posting_hash(job_posting: str, model: str = None) -> str:
    """
    Generate a hash for the job posting and model to use as cache key.
    Different models may generate different rubrics, so cache separately.
    
    Args:
        job_posting: The job posting text
        model: The model name (optional, uses default if not provided)
        
    Returns:
        SHA256 hash of the job posting + model
    """
    # Include model in the cache key to separate rubrics by model
    cache_input = f"{job_posting}||MODEL:{model or OPENROUTER_MODEL}"
    return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()[:16]


def load_rubric_from_cache(job_posting: str, model: str = None) -> Optional[EvaluationRubric]:
    """
    Load rubric from cache if it exists for this job posting and model.
    
    Args:
        job_posting: The job posting text
        model: The model name (optional, uses default if not provided)
        
    Returns:
        EvaluationRubric if cached, None otherwise
    """
    if not ENABLE_CACHE:
        return None
    
    cache_key = get_job_posting_hash(job_posting, model)
    cache_file = CACHE_DIR / f"rubric_{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_model = cached_data.get('model', 'unknown')
                print(f"✓ Loaded rubric from cache (key: {cache_key}, model: {cached_model})")
                return cached_data['rubric']
        except Exception as e:
            print(f"⚠ Cache load failed: {e}")
            return None
    
    return None


def save_rubric_to_cache(job_posting: str, rubric: EvaluationRubric, model: str = None):
    """
    Save rubric to cache with model-specific key.
    
    Args:
        job_posting: The job posting text
        rubric: The rubric to cache
        model: The model name (optional, uses default if not provided)
    """
    if not ENABLE_CACHE:
        return
    
    model_used = model or OPENROUTER_MODEL
    cache_key = get_job_posting_hash(job_posting, model_used)
    cache_file = CACHE_DIR / f"rubric_{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'job_posting': job_posting,
                'rubric': rubric,
                'model': model_used
            }, f)
        print(f"✓ Saved rubric to cache (key: {cache_key}, model: {model_used})")
    except Exception as e:
        print(f"⚠ Cache save failed: {e}")


def extract_rubric_with_llm(
    job_posting: str, 
    use_cache: bool = True,
    langfuse_parent=None,
    prompt_version: int = None,
    prompt_label: str = None,
    session_id: str = None,
    model: str = None
) -> EvaluationRubric:
    """
    Extract rubric from job posting using OpenRouter LLM API call.
    Uses cache to avoid re-generating rubric for the same job posting.
    
    LANGFUSE SPAN EXPLANATION:
    --------------------------
    A "span" is a logical operation within a trace.
    - Trace = Complete matching score calculation
    - Span = Sub-operation like "rubric extraction" or "criteria scoring"
    
    This creates a hierarchy:
    Trace: "Matching Score for Alice"
      ├─ Span: "Rubric Extraction"
      │   └─ Generation: "openrouter_call"
      └─ Span: "Criteria Scoring"
          └─ Generation: "openrouter_call"
    
    Benefits:
    - See which part is slow (rubric vs scoring)
    - Track success/failure of each operation
    - Measure cost per operation
    
    Args:
        job_posting: The job posting text
        use_cache: Whether to use cache (default: True)
        langfuse_trace: Parent trace for hierarchical tracking
        prompt_version: Specific Langfuse prompt version to use (e.g., 1, 2)
        prompt_label: Specific Langfuse prompt label to use (e.g., "production", "latest")
        
    Returns:
        EvaluationRubric with criteria and weights
    """
    # Try to load from cache first (cache is model-specific)
    if use_cache:
        cached_rubric = load_rubric_from_cache(job_posting, model)
        if cached_rubric is not None:
            return cached_rubric
    
    print(f"Job Posting: {job_posting[:200]}...")
    
    # LANGFUSE: Create span for this operation
    span = None
    if LANGFUSE_ENABLED and langfuse_parent:
        try:
            if hasattr(langfuse_parent, 'start_span'):
                span = langfuse_parent.start_span(
                    name="extract_rubric",
                    input={"job_posting_length": len(job_posting)},
                    metadata={"use_cache": use_cache}
                )
        except Exception as e:
            print(f"⚠ Langfuse span creation failed: {e}")
    
    # Prepare prompt (try Langfuse first, fallback to hardcoded)
    prompt_content = None
    langfuse_prompt = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Fetch specific version or label if provided
            if prompt_version:
                langfuse_prompt = langfuse.get_prompt("rubric-extraction", version=prompt_version)
                print(f"✓ Used managed prompt: 'rubric-extraction' (version {prompt_version})")
            elif prompt_label:
                langfuse_prompt = langfuse.get_prompt("rubric-extraction", label=prompt_label)
                print(f"✓ Used managed prompt: 'rubric-extraction' (label: {prompt_label})")
            else:
                langfuse_prompt = langfuse.get_prompt("rubric-extraction")
                print("✓ Used managed prompt: 'rubric-extraction' (latest)")
            
            # Compile with variables
            prompt_content = langfuse_prompt.compile(job_posting=job_posting)
        except Exception as e:
            print(f"⚠ Failed to fetch prompt from Langfuse: {e}")
            prompt_content = None
            langfuse_prompt = None
    
    # Fallback to hardcoded prompt
    if not prompt_content:
        prompt_content = f"{RUBRIC_EXTRACTION_PROMPT}\n\nJob Posting:\n{job_posting}"
        print("✓ Used fallback hardcoded prompt")

    try:
        # Call OpenRouter (returns content and LLM duration)
        response_text, llm_duration = call_openrouter(
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            max_tokens=4000,
            generation_name="rubric_extraction_llm",
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        
        print(f"✓ Rubric extraction LLM call: {llm_duration:.2f}s")
        
        print(f"LLM Response (first 500 chars): {response_text[:500]}...")
        print(f"LLM Response length: {len(response_text)} chars")
        
        # Clean and extract JSON from response (handles <think> blocks and markdown)
        response_text_original = response_text
        response_text = clean_llm_json_response(response_text)
        
        # Validate we have something to parse
        if not response_text or response_text == "{}":
            return EvaluationRubric(criteria=[], total_weight=0)
        
        # Parse JSON with better error message
        try:
            rubric_data = json.loads(response_text)
            if not isinstance(rubric_data, dict):
                raise ValueError(f"Expected dict from JSON, got {type(rubric_data)}")
            if "criteria" not in rubric_data:
                # Try fallback key
                if "criteria_scores" in rubric_data:
                    rubric_data["criteria"] = rubric_data["criteria_scores"]
                else:
                    return EvaluationRubric(criteria=[], total_weight=0)
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing failed at position {e.pos}: {e.msg}\n"
            error_msg += f"Cleaned response text (first 1000 chars):\n{response_text[:1000]}\n"
            error_msg += f"Original response (first 500 chars):\n{response_text_original[:500]}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg) from e
        
        # Convert to EvaluationRubric
        criteria = []
        raw_criteria = rubric_data.get("criteria") or []
        if not isinstance(raw_criteria, list):
            raw_criteria = []
            
        for c in raw_criteria:
            if not isinstance(c, dict): continue
            try:
                criteria.append(RubricCriterion(
                    name=str(c.get("name", "Unnamed")),
                    weight=float(c.get("weight", 0)),
                    description=str(c.get("description", "")),
                    is_required=bool(c.get("is_required", True))
                ))
            except:
                continue
        
        # Normalize weights to 100%
        total_weight = sum(c.weight for c in criteria)
        for criterion in criteria:
            criterion.weight = (criterion.weight / total_weight) * 100
        
        rubric = EvaluationRubric(criteria=criteria, total_weight=100.0)
        
        print(f"✓ Extracted {len(criteria)} criteria via LLM")
        
        # LANGFUSE: Span updated/closed automatically
        
        # Save to cache (model-specific)
        if use_cache:
            save_rubric_to_cache(job_posting, rubric, model)
        
        return rubric
        
    except Exception as e:
        print(f"ERROR in LLM call: {e}")
        # LANGFUSE: Error captured automatically by @observe
        raise


def score_criteria_with_llm(
    cv_profile: str, 
    rubric: EvaluationRubric,
    langfuse_parent=None,
    session_id: str = None,
    model: str = None
) -> List[CriterionScore]:
    """
    Score candidate against rubric criteria using OpenRouter LLM API call.
    
    Args:
        cv_profile: The candidate's CV text
        rubric: The evaluation rubric
        langfuse_trace: Parent trace for hierarchical tracking
        
    Returns:
        List of criterion scores
    """
    print(f"CV Profile: {cv_profile[:200]}...")
    print(f"Rubric: {len(rubric.criteria)} criteria")
    
    # Build rubric summary for prompt
    rubric_text = "\n".join([
        f"- {c.name} (Weight: {c.weight:.1f}%): {c.description}"
        for c in rubric.criteria
    ])
    
    # Debug: Print rubric to verify it's correct
    print(f"📋 Rubric being sent to LLM ({len(rubric.criteria)} criteria):")
    for i, criterion in enumerate(rubric.criteria, 1):
        print(f"  {i}. {criterion.name} (Weight: {criterion.weight:.1f}%)")
    print(f"📋 Rubric text preview (first 500 chars):\n{rubric_text[:500]}")
    
    # LANGFUSE: Trace/Span created automatically via @observe
    
    # Prepare prompt (try Langfuse first, fallback to hardcoded)
    prompt_content = None
    langfuse_prompt = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            langfuse_prompt = langfuse.get_prompt("criteria-scoring")
            print(f"✓ Fetched Langfuse prompt: 'criteria-scoring'")
            
            # Debug: Check if rubric_text variable exists in prompt template
            prompt_template = langfuse_prompt.prompt if hasattr(langfuse_prompt, 'prompt') else str(langfuse_prompt)
            if "{rubric_text}" not in prompt_template and "{{rubric_text}}" not in prompt_template:
                print(f"⚠⚠⚠ CRITICAL WARNING: Langfuse prompt template does NOT contain 'rubric_text' variable!")
                print(f"   The LLM will NOT receive the rubric criteria!")
                print(f"   Prompt template preview: {prompt_template[:500]}")
                print(f"   Falling back to hardcoded prompt to ensure rubric is included.")
                langfuse_prompt = None
                prompt_content = None
            else:
                # Compile with variables
                try:
                    prompt_content = langfuse_prompt.compile(
                        rubric_text=rubric_text,
                        cv_profile=cv_profile
                    )
                    print("✓ Compiled Langfuse prompt with rubric_text and cv_profile")
                    # print(f"======= Prompt content: {prompt_content}")
                    # Check if compilation actually worked (variables were substituted)
                    if "{rubric_text}" in prompt_content or "{{rubric_text}}" in prompt_content:
                        print("⚠⚠⚠ CRITICAL: Langfuse compile() did NOT substitute variables!")
                        print("   Placeholders still present in compiled prompt.")
                        print("   This might be a Langfuse SDK version issue or template format issue.")
                        print("   Trying manual substitution...")
                        
                        # Manual substitution as fallback
                        prompt_content = prompt_template.replace("{rubric_text}", rubric_text)
                        prompt_content = prompt_content.replace("{{rubric_text}}", rubric_text)
                        prompt_content = prompt_content.replace("{cv_profile}", cv_profile)
                        prompt_content = prompt_content.replace("{{cv_profile}}", cv_profile)
                        print("✓ Applied manual variable substitution")
                except Exception as e:
                    print(f"⚠ Langfuse compile() failed: {e}")
                    print("   Trying manual substitution...")
                    prompt_content = prompt_template.replace("{rubric_text}", rubric_text)
                    prompt_content = prompt_content.replace("{{rubric_text}}", rubric_text)
                    prompt_content = prompt_content.replace("{cv_profile}", cv_profile)
                    prompt_content = prompt_content.replace("{{cv_profile}}", cv_profile)
                    print("✓ Applied manual variable substitution")
                
                # Debug: Verify rubric is actually in compiled prompt
                rubric_check_passed = False
                if rubric_text[:200] in prompt_content:
                    rubric_check_passed = True
                    print(f"✓ Verified: Rubric text is present in compiled prompt")
                else:
                    # Check if criterion names are present
                    found_criteria = 0
                    missing_criteria = []
                    for criterion in rubric.criteria:
                        if criterion.name in prompt_content:
                            found_criteria += 1
                        else:
                            missing_criteria.append(criterion.name)
                    
                    if found_criteria >= len(rubric.criteria) * 0.8:  # At least 80% of criteria found
                        rubric_check_passed = True
                        print(f"✓ Verified: {found_criteria}/{len(rubric.criteria)} criterion names found in compiled prompt")
                    else:
                        print(f"⚠⚠⚠ WARNING: Only {found_criteria}/{len(rubric.criteria)} criterion names found in compiled prompt!")
                        print(f"   Missing criteria: {missing_criteria[:5]}")  # Show first 5 missing
                        print(f"   This suggests the rubric_text variable may not be properly included.")
                        print(f"   DEBUG: Compiled prompt length: {len(prompt_content)} chars")
                        print(f"   DEBUG: Rubric_text length: {len(rubric_text)} chars")
                        # Show where rubric_text appears (or doesn't) in the compiled prompt
                        if "Rubric:" in prompt_content or "RUBRIC:" in prompt_content or "**Rubric:**" in prompt_content:
                            rubric_idx = max(
                                prompt_content.find("Rubric:"),
                                prompt_content.find("RUBRIC:"),
                                prompt_content.find("**Rubric:**")
                            )
                            print(f"   DEBUG: Rubric section preview (500 chars after marker):")
                            print(f"   {prompt_content[rubric_idx:rubric_idx+500]}")
                        else:
                            print(f"   DEBUG: No 'Rubric:' marker found in compiled prompt!")
                            print(f"   DEBUG: Compiled prompt preview (first 1000 chars):")
                            print(f"   {prompt_content[:1000]}")
                        print(f"   Falling back to hardcoded prompt to ensure rubric is included.")
                        langfuse_prompt = None
                        prompt_content = None
                
        except Exception as e:
            print(f"⚠ Failed to fetch/compile Langfuse prompt: {e}")
            print(f"   Falling back to hardcoded prompt")
            prompt_content = None
            langfuse_prompt = None
            
    # Fallback to hardcoded prompt
    if not prompt_content:
        prompt_content = f"""{CRITERIA_SCORING_PROMPT}

## ⚠️ CRITICAL INSTRUCTION: USE ONLY THE PROVIDED CRITERIA ⚠️

**You MUST score ONLY the criteria listed below. DO NOT create new criteria or modify the criterion names.**

**Evaluation Criteria (YOU MUST SCORE EACH ONE):**
{rubric_text}

**Candidate CV:**
{cv_profile}

**CRITICAL REQUIREMENTS:**
1. **You MUST score EXACTLY {len(rubric.criteria)} criteria** - one for each criterion listed above
2. **Use the EXACT criterion names** as shown above (e.g., "{rubric.criteria[0].name if rubric.criteria else 'Criterion Name'}")
3. **DO NOT create new criteria** - only score the ones provided
4. **DO NOT combine or split criteria** - each criterion must be scored separately

**CRITICAL: You MUST return ONLY a valid JSON object.**
1. Use EXACT criterion name for "criteria_name" (e.g., "{rubric.criteria[0].name if rubric.criteria else 'Criterion Name'}"), NOT the weight.
2. "score" - number between 0-100.
3. "reasoning" - Step-by-step logic for the score calculation.
4. "evidence" - REQUIRED: specific evidence from the CV.
5. "gap" - REQUIRED if score < 80. Leave "" if score >= 80.

DO NOT include introductory text, concluding text, or markdown formatting outside the JSON block.
Return ONLY valid JSON with exactly the key "criteria_scores" containing the array of {len(rubric.criteria)} results."""
        print("✓ Used fallback hardcoded prompt")
    
    # Retry mechanism for JSON parsing failures
    max_retries = 3
    import time
    
    for attempt in range(max_retries):
        try:
            # Call OpenRouter (returns content and LLM duration)
            response_text, llm_duration = call_openrouter(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                max_tokens=6000,  # Increased to avoid truncation with reasoning
                generation_name="criteria_scoring_llm",
                langfuse_parent=langfuse_parent,
                langfuse_prompt=langfuse_prompt,
                session_id=session_id,
                model=model
            )
            
            print(f"✓ Criteria scoring LLM call (Attempt {attempt+1}/{max_retries}): {llm_duration:.2f}s")
            # print(f"LLM Response (first 500 chars): {response_text[:500]}...")
            print(f"LLM Response length: {len(response_text)} chars")
            
            # Clean and extract JSON from response (handles <think> blocks and markdown)
            response_text_original = response_text
            response_text = clean_llm_json_response(response_text)
            
            # Validate we have something to parse
            if not response_text or response_text == "{}":
                print(f"⚠ Warning: LLM returned empty JSON for criteria scoring. Original response: {response_text_original[:500]}")
                if attempt < max_retries - 1:
                    print("   Retrying...")
                    time.sleep(1)
                    continue
                return []
            
            # Parse JSON with better error message
            try:
                scores_data = json.loads(response_text)
                if not isinstance(scores_data, dict):
                    raise ValueError(f"Expected dict from JSON, got {type(scores_data)}")
                
                # If criteria_scores is missing, check for fallbacks
                if "criteria_scores" not in scores_data:
                    if "criteria" in scores_data:
                        scores_data["criteria_scores"] = scores_data["criteria"]
                    elif "scores" in scores_data:
                        scores_data["criteria_scores"] = scores_data["scores"]
                    else:
                        print(f"⚠ Warning: Missing 'criteria_scores' in response keys {list(scores_data.keys())}. Returning empty scores.")
                        if attempt < max_retries - 1:
                            print("   Retrying...")
                            time.sleep(1)
                            continue
                        return []
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed at position {e.pos}: {e.msg}\n"
                error_msg += f"Cleaned response text (first 1000 chars):\n{response_text[:1000]}\n"
                error_msg += f"Original response (first 500 chars):\n{response_text_original[:500]}"
                print(f"ERROR: {error_msg}")
                if attempt < max_retries - 1:
                    print("   Retrying...")
                    time.sleep(1)
                    continue
                raise ValueError(error_msg) from e
            
            # If we got here, we successfully parsed the JSON
            break
            
        except Exception as e:
            print(f"ERROR in LLM call (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("   Retrying...")
                time.sleep(1)
                continue
            # LANGFUSE: Error captured automatically by @observe
            raise

    # Debug: Print all returned criteria names
    returned_criteria_names = [s.get("criteria_name", "MISSING") for s in (scores_data.get("criteria_scores") or [])]
    expected_criteria_names = [c.name for c in rubric.criteria]
    print(f"\n{'='*80}")
    print(f"CRITERIA VALIDATION CHECK")
    print(f"{'='*80}")
    print(f"Expected criteria ({len(expected_criteria_names)}):")
    for i, name in enumerate(expected_criteria_names, 1):
        print(f"  {i}. {name}")
    print(f"\nReturned criteria ({len(returned_criteria_names)}):")
    for i, name in enumerate(returned_criteria_names, 1):
        match_indicator = "✓" if name in expected_criteria_names else "❌"
        print(f"  {i}. {match_indicator} {name}")
    print(f"{'='*80}\n")
    
    # Check for mismatches
    mismatches = []
    for returned_name in returned_criteria_names:
        if returned_name not in expected_criteria_names:
            # Try to find partial matches
            found_match = False
            for expected_name in expected_criteria_names:
                if expected_name.lower() in returned_name.lower() or returned_name.lower() in expected_name.lower():
                    print(f"⚠ Partial match found: '{returned_name}' might match '{expected_name}'")
                    found_match = True
                    break
            if not found_match:
                mismatches.append(returned_name)
    
    if mismatches:
        print(f"⚠ WARNING: {len(mismatches)} criteria returned that don't match the rubric:")
        for mismatch in mismatches:
            print(f"   - '{mismatch}' (not in rubric)")
        print(f"   Expected: {expected_criteria_names}")
        print(f"   This suggests the LLM may not have received the rubric properly or is generating its own criteria.")
    
    # Create a mapping from criterion names (with or without weight) to actual criterion names
    criterion_name_map = {}
    for criterion in rubric.criteria:
        # Map the exact name
        criterion_name_map[criterion.name] = criterion.name
        # Map name with weight format (as shown in prompt)
        criterion_name_map[f"{criterion.name} (Weight: {criterion.weight:.1f}%)"] = criterion.name
        # Map variations (case-insensitive, partial matches)
        criterion_name_map[criterion.name.lower()] = criterion.name
        # Try to match common variations
        if "frontend" in criterion.name.lower() or "front-end" in criterion.name.lower():
            criterion_name_map["Hard Skills - Front-end Technologies"] = criterion.name
            criterion_name_map["Front-end Technologies"] = criterion.name
        if "react" in criterion.name.lower():
            criterion_name_map["Hard Skills - React.js"] = criterion.name
        if "backend" in criterion.name.lower() or "back-end" in criterion.name.lower():
            # This might not be in rubric, but we'll try to match
            pass
    
    # Convert to CriterionScore list - ONLY for criteria that match the rubric
    scores = []
    matched_criteria = set()  # Track which rubric criteria have been matched
    
    for s in (scores_data.get("criteria_scores") or []):
        # Ensure all required fields exist
        if "criteria_name" not in s:
            print(f"WARNING: Missing 'criteria_name' in score: {s}")
            continue
        if "score" not in s:
            print(f"WARNING: Missing 'score' in score: {s}")
            continue
        
        # Normalize criteria name (remove weight if present)
        raw_criteria_name = s["criteria_name"]
        normalized_name = criterion_name_map.get(raw_criteria_name, raw_criteria_name)
        
        # If still not found, try to extract just the name part (before " (Weight:")
        if normalized_name == raw_criteria_name and " (Weight:" in raw_criteria_name:
            normalized_name = raw_criteria_name.split(" (Weight:")[0].strip()
            # Try to find matching criterion by name
            for criterion in rubric.criteria:
                if criterion.name == normalized_name:
                    criterion_name_map[raw_criteria_name] = normalized_name
                    break
        
        # Try fuzzy matching if exact match not found
        if normalized_name not in expected_criteria_names:
            # Try to find best match
            best_match = None
            best_similarity = 0
            for criterion in rubric.criteria:
                # Simple similarity check
                if normalized_name.lower() in criterion.name.lower() or criterion.name.lower() in normalized_name.lower():
                    similarity = min(len(normalized_name), len(criterion.name)) / max(len(normalized_name), len(criterion.name))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = criterion.name
            
            if best_match and best_similarity > 0.5:
                print(f"⚠ Fuzzy matched: '{raw_criteria_name}' -> '{best_match}' (similarity: {best_similarity:.2f})")
                normalized_name = best_match
            else:
                print(f"❌ ERROR: Criterion '{raw_criteria_name}' does not match any rubric criterion!")
                print(f"   Expected one of: {expected_criteria_names}")
                print(f"   Skipping this score to prevent incorrect matching.")
                continue
        
        # Debug: Log name normalization
        if raw_criteria_name != normalized_name:
            print(f"DEBUG: Normalized criteria name: '{raw_criteria_name}' -> '{normalized_name}'")
        
        # Check if we've already scored this criterion
        if normalized_name in matched_criteria:
            print(f"⚠ WARNING: Duplicate score for criterion '{normalized_name}'. Keeping first occurrence.")
            continue
        
        matched_criteria.add(normalized_name)
        
        # Clamp score to 0-100 range
        try:
            raw_score = float(s["score"])
        except (ValueError, TypeError):
            print(f"⚠ WARNING: Invalid score '{s.get('score')}'. Defaulting to 0.")
            raw_score = 0.0

        if raw_score < 0:
            print(f"⚠ WARNING: Score {raw_score} < 0. Clamping to 0.")
            raw_score = 0.0
        elif raw_score > 100:
            print(f"⚠ WARNING: Score {raw_score} > 100. Clamping to 100.")
            raw_score = 100.0
        
        score_obj = CriterionScore(
            criteria_name=normalized_name,  # Use normalized name
            score=raw_score,
            reasoning=s.get("reasoning", "") or "", # Added for stability
            evidence=s.get("evidence", "") or "",  # Ensure it's a string, not None
            gap=s.get("gap", "") or ""  # Ensure it's a string, not None
        )
        
        # Debug: Print if evidence/gap are empty
        if not score_obj.evidence and not score_obj.gap:
            print(f"WARNING: No evidence or gap for criterion: {score_obj.criteria_name}")
        
        scores.append(score_obj)
    
    # Return the processed scores
    print(f"✓ Returning {len(scores)} criteria scores")
    return scores


def generate_qualification_note(
    job_posting: str,
    cv_profile: str,
    rubric_text: str = None,
    criteria_scores_text: str = None,
    fit_level: str = None,
    language: str = "English",
    langfuse_parent=None,
    session_id: str = None,
    model: str = None
) -> str:
    """
    Generate a comprehensive qualification note for a candidate.
    
    Args:
        job_posting: The job posting text
        cv_profile: The candidate's CV text
        rubric_text: Formatted text of the evaluation rubric criteria
        criteria_scores_text: Formatted text of the criteria scores
        fit_level: The fit level assessment (e.g., "Strong Fit", "Good Fit")
        language: Language for the qualification note (default: "English")
        session_id: Optional session ID for Langfuse tracking
        model: Optional model name to use
        
    Returns:
        HTML-formatted qualification note
    """
    print(f"🌐 Language: {language}")
    if fit_level:
        print(f"🎯 Fit Level: {fit_level}")
    print(f"Qualification Note Generation for {len(cv_profile)} chars...")
    
    # Build structured context like the actual implementation
    base_context = "### INPUTS\n\n"
    
    # Fit Level Context (if provided) - CRITICAL: LLM must use this exact value
    if fit_level:
        base_context += "**FIT LEVEL ASSESSMENT (MANDATORY - DO NOT MODIFY):**\n"
        base_context += f"{fit_level}\n\n"
        base_context += "⚠️ **CRITICAL INSTRUCTION:** You MUST use the exact fit level '{fit_level}' in your assessment. "
        base_context += "Do NOT calculate or determine a different fit level. "
        base_context += "Start your response with: <b>OVERALL ASSESSMENT: {fit_level}</b>\n\n".format(fit_level=fit_level)
    
    # Rubric Context (if provided)
    if rubric_text:
        base_context += "**EVALUATION RUBRIC:**\n"
        base_context += f"{rubric_text}\n\n"
    
    # Criteria Scores Context (if provided)
    if criteria_scores_text:
        base_context += "**CRITERIA SCORES:**\n"
        base_context += f"{criteria_scores_text}\n\n"
    
    # Job Posting Context
    base_context += "**JOB POSTING:**\n"
    base_context += f"{job_posting}\n\n"
    
    # Profile Context
    base_context += "**CANDIDATE RÉSUMÉ:**\n"
    base_context += f"{cv_profile}\n\n"
    
    # Analysis Focus (Critical)
    base_context += """### ANALYSIS FOCUS (CRITICAL)

Before providing your qualification assessment, you MUST:

1. **IDENTIFY RECENT EXPERIENCE PATTERN** (Last 2-3 years):
   - What is their current role and responsibilities?
   - What type of work are they doing NOW?
   - What level/seniority are they operating at currently?

2. **DETECT CAREER TRAJECTORY**:
   - Are they moving UP in responsibility? (junior → senior → lead → manager)
   - Are they changing domains? (technical → business, IC → management)  
   - Is this job a natural NEXT STEP or a STEP BACK from their recent progression?

3. **ASSESS INTENT vs CAPABILITY**:
   - Do their recent 2-3 years suggest they WANT this type of role?
   - Or do they have old experience that shows they CAN do it but have moved away from it?

### MATCHING RULES TO APPLY:

- ❌ DON'T match based on skills from 5+ years ago if not used recently
- ❌ DON'T recommend someone for a role that's clearly below their recent level
- ✅ DO prioritize evidence from their last 2-3 years of work
- ✅ DO consider if this aligns with their career direction

Please assess this candidate with heavy emphasis on recent experience patterns and career trajectory alignment.

### LANGUAGE REQUIREMENT:
**IMPORTANT:** Generate the qualification note in **{language}**. All content, including headings, assessments, and recommendations, must be written in {language}.

### TASK:
**You have received all the required information above.** The job posting and candidate résumé have been provided. 

DO NOT ask for documents. DO NOT wait for input. 

**GENERATE THE COMPLETE QUALIFICATION NOTE NOW** following the exact HTML structure specified in your instructions, written entirely in **{language}**.

{fit_instruction}
""".format(
        language=language,
        fit_instruction=f"Start your response directly with:\n<b>OVERALL ASSESSMENT: {fit_level}</b>" if fit_level else "Start your response directly with:\n<b>OVERALL ASSESSMENT: [Fit Level]</b>"
    )
    
    # Prepare full prompt - System prompt first, then inputs
    prompt_content = f"""{QUALIFICATION_GENERATION_PROMPT}

---

{base_context}"""
    
    print(f"📋 Context length: {len(base_context)} chars")
    print(f"📋 Full prompt content length: {len(prompt_content)} chars")
    print(f"📋 Job posting preview (first 100 chars): {job_posting[:100]}")
    print(f"📋 CV profile preview (first 100 chars): {cv_profile[:100]}")
    
    # Try to use managed prompt from Langfuse
    langfuse_prompt = None
    prompt_to_use = prompt_content  # Default to full prompt (system + context)
    
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Get managed prompt from Langfuse (for tracking only)
            langfuse_prompt = langfuse.get_prompt("candidate-qualification")
            print(f"✓ Fetched managed prompt: 'candidate-qualification' (version: {langfuse_prompt.version}) - for tracking only")
            
            # Compile the prompt with variables
            compile_params = {
                "job_posting": job_posting,
                "cv_profile": cv_profile
            }
            if fit_level:
                compile_params["fit_level"] = fit_level
            if rubric_text:
                compile_params["rubric_text"] = rubric_text
            if criteria_scores_text:
                compile_params["criteria_scores_text"] = criteria_scores_text
            compile_params["language"] = language
            
            compiled = langfuse_prompt.compile(**compile_params)
            
            # Verify that the compiled prompt actually contains the actual data (not just templates)
            # Check if job posting content appears in the compiled prompt
            job_preview = job_posting[:100] if len(job_posting) > 100 else job_posting
            cv_preview = cv_profile[:100] if len(cv_profile) > 100 else cv_profile
            
            if job_preview in compiled and cv_preview in compiled:
                prompt_to_use = compiled
                print(f"✓ Using compiled Langfuse prompt (length: {len(compiled)} chars)")
                print(f"✓ Verified: Job posting and CV data present in compiled prompt")
            else:
                print("⚠ Langfuse prompt doesn't contain actual data - using local prompt with data")
                print(f"   Langfuse prompt length: {len(compiled)} chars")
                print(f"   Local prompt length: {len(prompt_content)} chars")
                prompt_to_use = prompt_content
            
        except Exception as e:
            print(f"⚠ Could not fetch Langfuse prompt, using local version: {e}")
            print("✓ Using local structured qualification prompt with full instructions")
    else:
        print("✓ Using local structured qualification prompt with full instructions")
    
    # Debug: Show what we're sending
    print(f"📤 Sending prompt to LLM (length: {len(prompt_to_use)} chars)")
    print(f"📝 Prompt preview (first 500 chars): {prompt_to_use[:500]}")
    
    try:
        # Call OpenRouter (returns content and LLM duration)
        response_text, llm_duration = call_openrouter(
            messages=[
                {
                    "role": "user",
                    "content": prompt_to_use
                }
            ],
            max_tokens=3000,
            generation_name="qualification_generation",
            langfuse_parent=langfuse_parent,
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        # print(f"✓ Generated qualification note : {response_text[:200]}")
        
        print(f"✓ Generated qualification note  ({len(response_text)} chars, LLM: {llm_duration:.2f}s)")
        
        return response_text
        
    except Exception as e:
        print(f"❌ Qualification generation failed: {e}")
        raise


def generate_qualification_summary(
    qualification_note: str,
    fit_level: str = None,
    language: str = "English",
    langfuse_parent=None,
    session_id: str = None,
    model: str = None,
    prompt_version: int = None,
    prompt_label: str = None
) -> str:
    """
    Generate a concise summary of the qualification note.
    
    Args:
        qualification_note: The full qualification note HTML text
        fit_level: The fit level assessment (e.g., "Strong Fit", "Good Fit")
        language: Language for the summary (default: "English")
        session_id: Optional session ID for Langfuse tracking
        model: Optional model name to use
        prompt_version: Optional specific Langfuse prompt version
        prompt_label: Optional Langfuse prompt label (e.g., 'production', 'latest')
        
    Returns:
        Concise summary text
    """
    print("\n[LLM CALL] Qualification Summary Generation...")
    print(f"🌐 Language: {language}")
    if fit_level:
        print(f"🎯 Fit Level (enforced): {fit_level}")
    
    # Build prompt for summary generation using the imported template
    summary_prompt_content = None
    langfuse_prompt = None
    
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Fetch from Langfuse with version/label if provided
            if prompt_version:
                langfuse_prompt = langfuse.get_prompt("qualification-summary", version=prompt_version)
                print(f"✓ Used Langfuse prompt: 'qualification-summary' (version {prompt_version})")
            elif prompt_label:
                langfuse_prompt = langfuse.get_prompt("qualification-summary", label=prompt_label)
                print(f"✓ Used Langfuse prompt: 'qualification-summary' (label: {prompt_label})")
            else:
                langfuse_prompt = langfuse.get_prompt("qualification-summary")
                print("✓ Used Langfuse prompt: 'qualification-summary' (production/default)")
            
            summary_prompt_content = langfuse_prompt.compile(
                qualification_note=qualification_note, 
                language=language
            )
        except Exception as e:
            print(f"⚠ Could not fetch 'qualification-summary' from Langfuse, using local fallback: {e}")
            summary_prompt_content = None
    
    # Fallback to local prompt if Langfuse failed or is disabled
    if not summary_prompt_content:
        summary_prompt_content = QUALIFICATION_SUMMARY.format(
            qualification_note=qualification_note, 
            language=language
        )
        print("✓ Used local fallback prompt for qualification summary")
    
    try:
        # Call OpenRouter (returns content and LLM duration)
        response_text, llm_duration = call_openrouter(
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt_content
                }
            ],
            max_tokens=500,  # Shorter for summary
            generation_name="qualification_summary",
            langfuse_parent=langfuse_parent,
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        
        print(f"✓ Generated qualification summary ({len(response_text)} chars, LLM: {llm_duration:.2f}s)")
        
        return response_text.strip()
        
    except Exception as e:
        print(f"❌ Qualification summary generation failed: {e}")
        raise


def pretty_print_results(rubric: EvaluationRubric, criteria_scores: List[CriterionScore], result: dict):
    """Pretty print the matching score results."""
    print("\n" + "="*100)
    print("EVALUATION RUBRIC (Job Requirements)")
    print("="*100)
    for criterion in rubric.criteria:
        required = "✓ REQUIRED" if criterion.is_required else "○ PREFERRED"
        print(f"\n{required} | {criterion.name} (Weight: {criterion.weight:.1f}%)")
        print(f"   Description: {criterion.description}")
    print(f"\nTotal Weight: {rubric.total_weight}%")
    print("="*100)
    
    print("\n" + "="*100)
    print("CRITERIA SCORES (Candidate Evaluation)")
    print("="*100)
    for item in result["breakdown"]:
        print(f"\n{item['criterion']}: {item['score']}/100 (Weight: {item['weight']:.1f}%)")
        print(f"   Contribution to final score: {item['contribution']:.2f} points")
        print(f"   Evidence: {item['evidence']}")
        if item['gap']:
            print(f"   Gap: {item['gap']}")
    
    print("\n" + "="*100)
    print(f"FINAL MATCHING SCORE: {result['final_score']}/100")
    print(f"Total Weight Used: {result['total_weight_used']:.1f}%")
    print("="*100 + "\n")


def clear_rubric_cache():
    """Clear all cached rubrics."""
    if CACHE_DIR.exists():
        cache_files = list(CACHE_DIR.glob("rubric_*.pkl"))
        for cache_file in cache_files:
            cache_file.unlink()
        print(f"✓ Cleared {len(cache_files)} cached rubric(s)")
    else:
        print("✓ No cache to clear")


def list_cached_rubrics():
    """List all cached rubrics."""
    if not CACHE_DIR.exists():
        print("No cached rubrics found")
        return
    
    cache_files = list(CACHE_DIR.glob("rubric_*.pkl"))
    if not cache_files:
        print("No cached rubrics found")
        return
    
    print(f"\nFound {len(cache_files)} cached rubric(s):")
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_key = cache_file.stem.replace("rubric_", "")
                job_preview = cached_data['job_posting'][:100]
                num_criteria = len(cached_data['rubric'].criteria)
                print(f"  - {cache_key}: {num_criteria} criteria | {job_preview}...")
        except Exception as e:
            print(f"  - {cache_file.name}: [Error reading cache: {e}]")


def test_scenario(scenario_name: str, job_posting: str, cv_profile: str, use_cache: bool = True):
    """
    Run a test scenario with job posting and CV profile using actual LLM calls.
    
    LANGFUSE TRACE EXPLANATION:
    ---------------------------
    A "trace" is the top-level container for a complete operation.
    Think of it as a "session" or "request".
    
    Hierarchy:
    Trace: "Matching Score - Alice vs Senior Dev"
      ├─ Span: "Rubric Extraction"
      │   └─ Generation: "rubric_extraction_llm" (LLM call)
      └─ Span: "Criteria Scoring"
          └─ Generation: "criteria_scoring_llm" (LLM call)
    
    Benefits:
    - See the complete flow of one matching score calculation
    - Track total time and cost for the entire operation
    - Add scores/feedback at the trace level (quality metrics)
    - Filter and search traces by name, user, session, etc.
    
    Args:
        scenario_name: Name of the test scenario
        job_posting: The job posting text
        cv_profile: The candidate's CV text
        use_cache: Whether to use cached rubric if available (default: True)
    """
    print("\n" + "#"*100)
    print(f"TEST SCENARIO: {scenario_name}")
    print("#"*100)
    print(f"\nJob Posting:\n{job_posting[:300]}...")
    print(f"\nCV Profile:\n{cv_profile[:300]}...")
    
    # LANGFUSE: Create a trace for this matching score calculation
    trace = None
    if LANGFUSE_ENABLED:
        # A trace is the top-level unit of observation
        # It represents one complete matching score calculation
        trace = langfuse.trace(
            name=f"matching_score",
            user_id=f"scenario_{scenario_name.lower().replace(' ', '_')}",
            metadata={
                "scenario_name": scenario_name,
                "use_cache": use_cache,
                "job_posting_preview": job_posting[:200],
                "cv_profile_preview": cv_profile[:200]
            },
            tags=["test", "matching_score"]
        )
    
    # Step 1: Extract rubric from job posting (uses cache if available)
    rubric = extract_rubric_with_llm(job_posting, use_cache=use_cache, langfuse_trace=trace)
    
    # Step 2: Score candidate against rubric (ACTUAL LLM CALL)
    criteria_scores = score_criteria_with_llm(cv_profile, rubric, langfuse_trace=trace)
    
    # Step 3: Calculate final matching score (actual calculation logic)
    result = calculate_matching_score(rubric, criteria_scores)
    
    # LANGFUSE: Add the final score as a "score" metric
    # This allows you to:
    # 1. Track quality metrics over time
    # 2. Filter traces by score ranges
    # 3. Compare different prompt versions by their scores
    if LANGFUSE_ENABLED and trace:
        # Add the matching score as a metric
        trace.score(
            name="matching_score",
            value=result["final_score"],
            comment=f"Weighted average of {len(result['breakdown'])} criteria"
        )
        
        # Add quality indicators
        if result["final_score"] >= 80:
            trace.score(name="quality", value=1, comment="Strong match")
        elif result["final_score"] >= 60:
            trace.score(name="quality", value=0.7, comment="Good match")
        elif result["final_score"] >= 40:
            trace.score(name="quality", value=0.5, comment="Moderate match")
        else:
            trace.score(name="quality", value=0.3, comment="Weak match")
    
    # Pretty print results
    pretty_print_results(rubric, criteria_scores, result)
    
    return result


def test_multiple_profiles(job_posting: str, cv_profiles: List[tuple], use_cache: bool = True):
    """
    Test multiple candidate profiles against the same job posting.
    Uses cached rubric to ensure consistency across all candidates.
    
    Args:
        job_posting: The job posting text
        cv_profiles: List of (name, cv_text) tuples
        use_cache: Whether to use cached rubric (default: True)
        
    Returns:
        List of (name, score, result) tuples
    """
    print("\n" + "="*100)
    print(f"TESTING {len(cv_profiles)} CANDIDATES AGAINST SAME JOB POSTING")
    print("="*100)
    
    # Extract rubric once (will be cached)
    print("\n[EXTRACTING RUBRIC]")
    
    # LANGFUSE: Create a parent trace for the batch comparison
    batch_trace = None
    if LANGFUSE_ENABLED:
        batch_trace = langfuse.trace(
            name="batch_matching_scores",
            metadata={
                "num_candidates": len(cv_profiles),
                "use_cache": use_cache,
                "job_posting_preview": job_posting[:200]
            },
            tags=["test", "batch", "comparison"]
        )
    
    rubric = extract_rubric_with_llm(job_posting, use_cache=use_cache, langfuse_trace=batch_trace)
    
    results = []
    for name, cv_profile in cv_profiles:
        print(f"\n{'='*100}")
        print(f"CANDIDATE: {name}")
        print(f"{'='*100}")
        print(f"CV: {cv_profile[:200]}...")
        
        # LANGFUSE: Create a sub-trace for each candidate
        # This allows you to compare candidates side-by-side
        candidate_trace = None
        if LANGFUSE_ENABLED:
            candidate_trace = langfuse.trace(
                name=f"matching_score",
                user_id=name.lower().replace(" ", "_"),
                metadata={
                    "candidate_name": name,
                    "cv_preview": cv_profile[:200]
                },
                tags=["batch", "candidate", name.lower()]
            )
        
        # Score this candidate
        criteria_scores = score_criteria_with_llm(cv_profile, rubric, langfuse_trace=candidate_trace)
        result = calculate_matching_score(rubric, criteria_scores)
        
        # LANGFUSE: Add score to candidate trace
        if LANGFUSE_ENABLED and candidate_trace:
            candidate_trace.score(
                name="matching_score",
                value=result["final_score"],
                comment=f"Candidate: {name}"
            )
        
        # Store result
        results.append((name, result['final_score'], result))
        
        # Print summary
        print(f"\n→ {name}: {result['final_score']}/100")
    
    # Print comparison summary
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (name, score, _) in enumerate(results, 1):
        print(f"{i}. {name}: {score}/100")
    print("="*100)
    
    return results


def main():
    """Main test function with various scenarios."""
    
    # Test Scenario 1: Perfect Senior Match
    job_posting_1 = """
    Senior Front-end Developer
    
    We are looking for a Senior Front-end Developer with 5+ years of experience to join our team.
    
    Requirements:
    - 5+ years of professional React.js development experience
    - Expert-level JavaScript, HTML, and CSS skills
    - Strong communication and collaboration skills
    - Professional proficiency in French and English
    
    Nice to have:
    - Experience with Next.js, Jest, or Cypress
    """
    
    cv_profile_1 = """
    Alice Johnson
    Senior Front-end Developer
    
    Professional Experience: 6 years in front-end development
    
    Skills:
    - React.js: 6 years (Senior level, led multiple large-scale projects)
    - JavaScript: 7 years (Expert level)
    - HTML/CSS: 7 years (Expert level)
    - Next.js: 2 years
    - Jest: 3 years
    
    Languages:
    - English: Native
    - French: Professional (C1)
    
    Summary: Experienced senior developer with strong team collaboration skills. Led multiple cross-functional teams in multicultural environments.
    """
    
    test_scenario("Perfect Senior Match", job_posting_1, cv_profile_1)
    
    # Test Scenario 2: Junior Applying for Senior (Should score LOW)
    job_posting_2 = """
    Senior Front-end Developer
    
    We are looking for a Senior Front-end Developer with 5+ years of experience.
    
    Requirements:
    - 5+ years of professional React.js development experience
    - Expert-level JavaScript, HTML, and CSS skills
    - Professional French and English
    """
    
    cv_profile_2 = """
    Bob Smith
    Junior Front-end Developer
    
    Professional Experience: 7 months in front-end development
    
    Skills:
    - React.js: 7 months (Junior level)
    - JavaScript: 10 months
    - HTML/CSS: 1 year
    
    Languages:
    - English: Fluent
    - French: Professional (B2)
    
    Summary: Enthusiastic junior developer eager to learn and grow. Quick learner with strong problem-solving skills.
    """
    
    # test_scenario("Junior Applying for Senior (Underqualified)", job_posting_2, cv_profile_2)
    
    # Test Scenario 3: Medior with Good Skills (Should score MEDIUM-HIGH)
    job_posting_3 = """
    Senior React Developer
    
    Requirements:
    - 5+ years React.js experience
    - Strong JavaScript, HTML, CSS
    - French and English proficiency
    """
    
    cv_profile_3 = """
    Carol Davis
    Mid-level Front-end Developer
    
    Professional Experience: 4 years in front-end development
    
    Skills:
    - React.js: 3 years (Medior level)
    - JavaScript: 4 years
    - HTML/CSS: 4 years
    - Cypress: 1 year
    
    Languages:
    - English: Fluent
    - French: Advanced (C1)
    
    Summary: Solid team player with good technical skills. Strong communication and collaboration abilities.
    """
    
    # test_scenario("Medior with Good Skills (Partially Qualified)", job_posting_3, cv_profile_3)
    
    # Test Scenario 4: Senior without Required French Language
    job_posting_4 = """
    Senior React Developer
    
    Requirements:
    - 5+ years React.js experience
    - Professional French (required for client communication)
    - English proficiency
    """
    
    cv_profile_4 = """
    David Lee
    Senior Front-end Developer
    
    Professional Experience: 7 years
    
    Skills:
    - React.js: 7 years (Senior level, architect-level expertise)
    - JavaScript: 8 years
    - Next.js: 4 years
    - Jest, Cypress: 5 years
    
    Languages:
    - English: Native
    
    Summary: Highly skilled senior developer with extensive experience in modern front-end technologies and team leadership.
    """
    
    # test_scenario("Senior without French Language (Missing Critical Requirement)", job_posting_4, cv_profile_4)
    
    # Test Scenario 5: Senior with Wrong Tech Stack
    job_posting_5 = """
    Senior React Developer
    
    Requirements:
    - 5+ years React.js experience (required)
    - Expert JavaScript
    - French and English
    """
    
    cv_profile_5 = """
    Eva Martinez
    Senior Front-end Developer
    
    Professional Experience: 8 years in frontend development
    
    Skills:
    - Vue.js: 6 years (Senior level)
    - Angular: 4 years (Senior level)
    - JavaScript: 8 years (Expert level)
    - HTML/CSS: 8 years
    
    Languages:
    - English: Native
    - French: Professional (C1)
    
    Summary: Experienced senior developer specializing in Vue.js and Angular frameworks. Strong team leadership and mentoring experience.
    """
    
    # test_scenario("Senior with Wrong Tech Stack (Missing React)", job_posting_5, cv_profile_5)
    
    print("\n" + "#"*100)
    print("ALL TESTS COMPLETED")
    print("#"*100 + "\n")


if __name__ == "__main__":
    # Run all test scenarios
    main()
    
    # ========================================================================
    # CACHE MANAGEMENT EXAMPLES
    # ========================================================================
    
    # List cached rubrics
    # list_cached_rubrics()
    
    # Clear cache (force regeneration of all rubrics)
    # clear_rubric_cache()
    
    # ========================================================================
    # SINGLE SCENARIO EXAMPLE
    # ========================================================================
    
    # job_posting = """
    # Your custom job posting here...
    # """
    # cv_profile = """
    # Your custom CV here...
    # """
    # test_scenario("My Custom Test", job_posting, cv_profile)
    
    # ========================================================================
    # MULTIPLE PROFILES EXAMPLE (uses cached rubric for consistency)
    # ========================================================================
    
    # job_posting = """
    # Senior Python Developer
    # 
    # Requirements:
    # - 5+ years Python experience
    # - Django or Flask expertise
    # - PostgreSQL proficiency
    # """
    # 
    # candidates = [
    #     ("Alice (Senior)", "Alice Johnson\nSenior Python Developer\n7 years Python, 5 years Django, PostgreSQL expert"),
    #     ("Bob (Junior)", "Bob Smith\nJunior Python Developer\n1 year Python, basic Django"),
    #     ("Carol (Medior)", "Carol Davis\nMid-level Python Developer\n4 years Python, 3 years Flask, PostgreSQL")
    # ]
    # 
    # results = test_multiple_profiles(job_posting, candidates)

