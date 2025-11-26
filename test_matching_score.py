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
from prompts import CRITERIA_SCORING_PROMPT, QUALIFICATION_GENERATION_PROMPT

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
    evidence: str
    gap: str


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get API key from environment variable or .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found!")
    print("Please set it in one of these ways:")
    print("  1. Create a .env file with: OPENROUTER_API_KEY=your-key-here")
    print("  2. Set environment variable: export OPENROUTER_API_KEY='your-key-here'")
    print("  3. Install python-dotenv: pip install python-dotenv")
    sys.exit(1)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available Models
CLAUDE_HAIKU_OPENROUTER = "anthropic/claude-haiku-4.5"
GEMINI_FLASH_OPENROUTER = "google/gemini-2.5-flash-preview-09-2025"
GEMINI_FLASH_LITE_OPENROUTER = "google/gemini-2.5-flash-lite"
GPT_OSS_120B_OPENROUTER = "openai/gpt-oss-120b:exacto"

# Default model (can be overridden)
OPENROUTER_MODEL = CLAUDE_HAIKU_OPENROUTER

# Model display names for UI
MODEL_NAMES = {
    CLAUDE_HAIKU_OPENROUTER: "Claude Haiku 4.5",
    GEMINI_FLASH_OPENROUTER: "Gemini 2.5 Flash Preview",
    GEMINI_FLASH_LITE_OPENROUTER: "Gemini 2.5 Flash Lite",
    GPT_OSS_120B_OPENROUTER: "GPT OSS 120B (Exacto)"
}

# Cache configuration
CACHE_DIR = Path(__file__).parent / ".rubric_cache"
CACHE_DIR.mkdir(exist_ok=True)
ENABLE_CACHE = True  # Set to False to disable caching

# ============================================================================
# PROMPTS (copied from actual project)
# ============================================================================

# CRITERIA_SCORING_PROMPT is imported from prompts.py at the top of the file

# Rubric extraction prompt (actual prompt used in project)
# RUBRIC_EXTRACTION_PROMPT = """You are an AI that extracts evaluation criteria from job postings.

# Given a job posting, extract 6-10 weighted criteria that should be used to evaluate candidates.

# **CRITICAL INSTRUCTIONS FOR TECHNICAL SKILLS:**
# 1. **Always include required level/seniority in the description**
#    - Example: "Strong React.js expertise (Senior level, 5+ years)"
#    - Example: "Python proficiency (Medior level, 3+ years)"
#    - Example: "Basic SQL knowledge (Junior level, 1+ years)"

# 2. **Weight criteria based on importance:**
#    - Critical must-haves: 15-25%
#    - Important requirements: 10-15%
#    - Nice-to-haves: 5-10%
#    - Total must equal 100%

# 3. **Categories to consider:**
#    - Job Title Match
#    - Experience Years
#    - Technical Skills (break down by technology, always include required level)
#    - Soft Skills
#    - Languages
#    - Industry Experience
#    - Education

# Return ONLY a valid JSON object with this structure (no markdown, no explanations):
# {{
#   "criteria": [
#     {{
#       "name": "Criterion Name",
#       "weight": 15.0,
#       "description": "Detailed description including required level (e.g., 'Senior level', '5+ years')",
#       "is_required": true
#     }}
#   ]
# }}

# Extract criteria now from the job posting below."""
RUBRIC_EXTRACTION_PROMPT = """You are an expert recruiter tasked with extracting evaluation criteria from a job posting.

## YOUR MISSION
Analyze the job posting and create a weighted scoring rubric with specific criteria that will be used to evaluate candidates.

## CRITERIA CATEGORIES

Extract criteria from these categories (only if relevant to the job):

1. **Job Title Match** (10-20 weight)
   - How well candidate's job titles align with the target role
   - More weight if specific title/role is critical

2. **Experience Years** (15-25 weight)
   - Years of relevant experience required
   - More weight for senior roles or specialized experience

3. **Seniority Level** (5-15 weight)
   - Leadership level, team management, strategic responsibility
   - More weight for leadership positions

4. **Technical Skills** (20-35 weight)
   - Specific technologies, tools, frameworks, methodologies
   - More weight for technical roles
   - Break down by: Required (must-have) vs Preferred (nice-to-have)
   - ⚠️ **CRITICAL:** Include the REQUIRED LEVEL in the description (e.g., "Expert React (5+ years)", "Senior Python developer", "Medior/Senior level frontend")

5. **Soft Skills** (5-15 weight)
   - Communication, teamwork, problem-solving, etc.
   - More weight if explicitly emphasized

6. **Industry Experience** (5-15 weight)
   - Specific industry background (Finance, Healthcare, etc.)
   - More weight if domain knowledge is critical

7. **Languages** (5-20 weight)
   - Required language proficiencies with levels
   - More weight if multilingual role or client-facing

8. **Location** (0-10 weight)
   - Geographic requirements, remote vs on-site
   - More weight if relocation is difficult or on-site is mandatory

9. **Education** (5-15 weight)
   - Required degrees, certifications
   - More weight for roles requiring specific qualifications

10. **Domain Knowledge** (5-15 weight)
    - Specific business domain expertise
    - More weight if niche knowledge required

## CRITICAL INSTRUCTION FOR DESCRIPTIONS

⚠️ **Each criterion description MUST include the required level/experience:**

**For Technical Skills:**
- Extract the required proficiency level from the job posting
- Include years if mentioned (e.g., "5+ years", "3+ years")
- Include seniority indicators (e.g., "Expert", "Senior level", "Medior/Senior", "Strong expertise")
- Examples:
  - ✅ "Expert React (5+ years, complex applications)"
  - ✅ "Senior Python developer (3+ years)"
  - ✅ "Medior/Senior level frontend development"
  - ❌ "React" (too vague, missing level)
  - ❌ "Python experience" (missing required level)

**For Experience Years:**
- Be explicit: "5+ years", "3-5 years", "8+ years"

**For Seniority Level:**
- Specify: "Senior level", "Medior/Senior", "Lead level", "Junior level"

## WEIGHTING GUIDELINES

- Total weights MUST sum to 100
- Assign higher weights to:
  - Explicitly stated "must-haves" or "required"
  - Skills/experience mentioned multiple times
  - Critical responsibilities in the role
  - Blocking requirements (without X, cannot do Y)
  
- Assign lower weights to:
  - "Nice to have" or "preferred"
  - Generic requirements
  - Easily trainable skills

## OUTPUT FORMAT

Return ONLY a valid JSON object:

```json
{{
  "criteria": [
    {{
      "name": "Experience Years",
      "weight": 20,
      "description": "5+ years in backend development",
      "is_required": true
    }},
    {{
      "name": "Technical Skills - Python",
      "weight": 15,
      "description": "Expert Python development",
      "is_required": true
    }},
    {{
      "name": "Technical Skills - AWS",
      "weight": 10,
      "description": "AWS cloud services experience",
      "is_required": true
    }},
    {{
      "name": "Languages - Dutch",
      "weight": 15,
      "description": "Professional Dutch (B2+ level)",
      "is_required": true
    }},
    {{
      "name": "Languages - English",
      "weight": 10,
      "description": "Fluent English",
      "is_required": true
    }},
    {{
      "name": "Industry - Financial Services",
      "weight": 10,
      "description": "Experience in banking/fintech",
      "is_required": false
    }},
    {{
      "name": "Soft Skills - Communication",
      "weight": 10,
      "description": "Strong stakeholder communication",
      "is_required": false
    }},
    {{
      "name": "Location",
      "weight": 5,
      "description": "Brussels-based or willing to relocate",
      "is_required": false
    }},
    {{
      "name": "Education",
      "weight": 5,
      "description": "Bachelor's degree in Computer Science or related",
      "is_required": false
    }}
  ],
  "total_weight": 100
}}
```

## VALIDATION RULES

1. Criteria names should be clear and specific
2. Each criterion MUST have a weight > 0
3. Sum of all weights MUST equal exactly 100
4. is_required should be true for critical/blocking requirements
5. **Description MUST include the required level/years** (e.g., "5+ years", "Expert level", "Senior", "Medior/Senior")
6. Description should be specific to this job (not generic)

## EXAMPLES

**Job Posting:** "Senior Backend Developer, 5+ years Python, AWS required, Dutch B2+"

**Good Rubric:**
- Experience Years (20) - "5+ years backend development"
- Technical Skills - Python (25) - "Expert Python (5+ years)"
- Technical Skills - AWS (20) - "AWS services (production experience)"
- Languages - Dutch (20) - "Dutch B2+ level"
- Languages - English (10) - "Professional English"
- Soft Skills (5) - "Communication skills"

**Job Posting:** "Medior/Senior Frontend Developer, React expertise, Next.js"

**Good Rubric:**
- Seniority Level (15) - "Medior or Senior level (3+ years)"
- Technical Skills - React (30) - "Strong React expertise (3+ years, complex applications)"
- Technical Skills - Next.js (15) - "Next.js experience (production use)"
- Technical Skills - JavaScript (20) - "Advanced JavaScript (ES6+, 3+ years)"

**Bad Rubric:**
- Technical Skills (50) - Too broad, should split by technology
- Technical Skills - React (30) - "React" ❌ (missing required level)
- Languages (10) - Not specific about which languages
- Generic (20) - Vague criteria name
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
            
            # Ensure evidence and gap are strings (not None)
            evidence = criterion_score.evidence if criterion_score.evidence else ""
            gap = criterion_score.gap if criterion_score.gap else ""
            
            breakdown.append({
                "criterion": criterion_score.criteria_name,
                "score": criterion_score.score,
                "weight": weight,
                "contribution": round(contribution, 2),
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
        final_score = round(sum(c.score for c in criteria_scores) / len(criteria_scores))
    
    return {
        "final_score": final_score,
        "total_weight_used": total_weight,
        "breakdown": breakdown
    }


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
    Make an API call to OpenRouter with Langfuse observability.
    
    LANGFUSE TRACING EXPLANATION:
    ------------------------------
    When LANGFUSE_ENABLED=True, this function:
    1. Creates a "generation" (LLM call) in Langfuse
    2. Logs: input (prompt), output (response), model, tokens, latency
    3. Links to parent trace/span (if provided)
    4. Calculates cost based on token usage
    
    This allows you to:
    - See all LLM calls in Langfuse dashboard
    - Compare different prompts/models
    - Track costs and performance
    - Debug issues by viewing exact inputs/outputs
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        langfuse_parent: Parent trace/span for hierarchical tracking
        generation_name: Name for this LLM call (e.g., "rubric_extraction", "criteria_scoring")
        
    Returns:
        Response text from the model
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/wiggli-parser",  # Optional
        "X-Title": "Wiggli Parser Test"  # Optional
    }
    
    # Use the provided model or fall back to default
    selected_model = model if model else OPENROUTER_MODEL
    
    data = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    # LANGFUSE: Create generation manually (v3.x API with session grouping)
    generation = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Build generation parameters
            gen_params = {
                "name": generation_name,
                "model": selected_model,
                "model_parameters": {"max_tokens": max_tokens},
                "input": messages,
                "prompt": langfuse_prompt
            }
            
            # Session ID will be added via update() after creation
            # Create generation using the client directly
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

    # Track actual LLM API call time (excluding Langfuse overhead)
    import time
    llm_start_time = time.time()
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=data)
    llm_duration = time.time() - llm_start_time
    
    # Check for HTTP errors
    if response.status_code != 200:
        error_detail = response.text
        try:
            error_json = response.json()
            error_detail = json.dumps(error_json, indent=2)
        except:
            pass
        # Update generation with error
        if generation:
            generation.end(level="ERROR", status_message=error_detail)
        raise ValueError(f"OpenRouter API error (status {response.status_code}): {error_detail}")
    
    response.raise_for_status()
    
    # Parse response
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        if generation:
            generation.end(level="ERROR", status_message=str(e))
        raise ValueError(f"Invalid JSON response from API: {response.text[:500]}") from e
    
    # Check for API-level errors in response
    if "error" in result:
        error_msg = result.get("error", {})
        if isinstance(error_msg, dict):
            error_detail = error_msg.get("message", str(error_msg))
        else:
            error_detail = str(error_msg)
        if generation:
            generation.end(level="ERROR", status_message=error_detail)
        raise ValueError(f"OpenRouter API error: {error_detail}")
    
    # Extract content
    if "choices" not in result or len(result["choices"]) == 0:
        error_msg = f"Unexpected API response format: {json.dumps(result, indent=2)[:500]}"
        if generation:
            generation.end(level="ERROR", status_message=error_msg)
        raise ValueError(error_msg)
    
    content = result["choices"][0]["message"]["content"]
    
    # Check if content is empty
    if not content or not content.strip():
        error_msg = f"Empty response from API. Full response: {json.dumps(result, indent=2)[:500]}"
        if generation:
            generation.end(level="ERROR", status_message=error_msg)
        raise ValueError(error_msg)
    
    # LANGFUSE: Update generation with output
    if generation:
        try:
            # Extract token usage
            usage = result.get("usage", {})
            # Update output and usage first
            generation.update(
                output=content,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            )
            # Then end the generation
            generation.end()
        except Exception as e:
            print(f"⚠ Langfuse generation update failed: {e}")
    
    # Return content and actual LLM call duration (excluding Langfuse overhead)
    print(f"⏱️  LLM API call took: {llm_duration:.2f}s")
    return content, llm_duration


def get_job_posting_hash(job_posting: str) -> str:
    """
    Generate a hash for the job posting to use as cache key.
    
    Args:
        job_posting: The job posting text
        
    Returns:
        SHA256 hash of the job posting
    """
    return hashlib.sha256(job_posting.encode('utf-8')).hexdigest()[:16]


def load_rubric_from_cache(job_posting: str) -> Optional[EvaluationRubric]:
    """
    Load rubric from cache if it exists.
    
    Args:
        job_posting: The job posting text
        
    Returns:
        EvaluationRubric if cached, None otherwise
    """
    if not ENABLE_CACHE:
        return None
    
    cache_key = get_job_posting_hash(job_posting)
    cache_file = CACHE_DIR / f"rubric_{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                print(f"✓ Loaded rubric from cache (key: {cache_key})")
                return cached_data['rubric']
        except Exception as e:
            print(f"⚠ Cache load failed: {e}")
            return None
    
    return None


def save_rubric_to_cache(job_posting: str, rubric: EvaluationRubric):
    """
    Save rubric to cache.
    
    Args:
        job_posting: The job posting text
        rubric: The rubric to cache
    """
    if not ENABLE_CACHE:
        return
    
    cache_key = get_job_posting_hash(job_posting)
    cache_file = CACHE_DIR / f"rubric_{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'job_posting': job_posting,
                'rubric': rubric,
                'model': OPENROUTER_MODEL
            }, f)
        print(f"✓ Saved rubric to cache (key: {cache_key})")
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
    # Try to load from cache first
    if use_cache:
        cached_rubric = load_rubric_from_cache(job_posting)
        if cached_rubric is not None:
            return cached_rubric
    
    print("\n[LLM CALL via OpenRouter] Rubric Extraction from Job Posting...")
    print(f"Model: {OPENROUTER_MODEL}")
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
            max_tokens=2000,
            generation_name="rubric_extraction_llm",
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        
        print(f"✓ Rubric extraction LLM call: {llm_duration:.2f}s")
        
        print(f"LLM Response (first 500 chars): {response_text[:500]}...")
        print(f"LLM Response length: {len(response_text)} chars")
        
        # Try to parse JSON (handle potential markdown wrapping and extra text)
        response_text_original = response_text
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if "```json" in response_text:
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        elif "```" in response_text:
            start_idx = response_text.find("```") + 3
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        
        # Try to find JSON object boundaries
        if "{" in response_text and "}" in response_text:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
        
        response_text = response_text.strip()
        
        # Validate we have something to parse
        if not response_text:
            raise ValueError(f"Empty response after parsing. Original response: {response_text_original[:500]}")
        
        # Parse JSON with better error message
        try:
            rubric_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing failed at position {e.pos}: {e.msg}\n"
            error_msg += f"Response text (first 1000 chars):\n{response_text[:1000]}\n"
            error_msg += f"Original response (first 500 chars):\n{response_text_original[:500]}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg) from e
        
        # Convert to EvaluationRubric
        criteria = [
            RubricCriterion(
                name=c["name"],
                weight=float(c["weight"]),
                description=c["description"],
                is_required=c.get("is_required", True)
            )
            for c in rubric_data["criteria"]
        ]
        
        # Normalize weights to 100%
        total_weight = sum(c.weight for c in criteria)
        for criterion in criteria:
            criterion.weight = (criterion.weight / total_weight) * 100
        
        rubric = EvaluationRubric(criteria=criteria, total_weight=100.0)
        
        print(f"✓ Extracted {len(criteria)} criteria via LLM")
        
        # LANGFUSE: Span updated/closed automatically
        
        # Save to cache
        if use_cache:
            save_rubric_to_cache(job_posting, rubric)
        
        return rubric
        
    except Exception as e:
        print(f"ERROR in LLM call: {e}")
        # LANGFUSE: Error captured automatically by @observe
        raise


def score_criteria_with_llm(
    cv_profile: str, 
    rubric: EvaluationRubric,
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
    print("\n[LLM CALL via OpenRouter] Criteria Scoring...")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"CV Profile: {cv_profile[:200]}...")
    print(f"Rubric: {len(rubric.criteria)} criteria")
    
    # Build rubric summary for prompt
    rubric_text = "\n".join([
        f"- {c.name} (Weight: {c.weight:.1f}%): {c.description}"
        for c in rubric.criteria
    ])
    
    # LANGFUSE: Trace/Span created automatically via @observe
    
    # Prepare prompt (try Langfuse first, fallback to hardcoded)
    prompt_content = None
    langfuse_prompt = None
    if LANGFUSE_ENABLED and langfuse:
        try:
            langfuse_prompt = langfuse.get_prompt("criteria-scoring")
            # Compile with variables
            prompt_content = langfuse_prompt.compile(
                rubric_text=rubric_text,
                cv_profile=cv_profile
            )
            print("✓ Used managed prompt: 'criteria-scoring'")
        except Exception as e:
            print(f"⚠ Failed to fetch prompt from Langfuse: {e}")
            prompt_content = None
            langfuse_prompt = None
            
    # Fallback to hardcoded prompt
    if not prompt_content:
        prompt_content = f"""{CRITERIA_SCORING_PROMPT}

**Evaluation Criteria:**
{rubric_text}

**Candidate CV:**
{cv_profile}

**CRITICAL: For EACH criterion, you MUST provide:**
1. "criteria_name" - Use ONLY the criterion name (e.g., "Seniority Level"), NOT the weight part (e.g., NOT "Seniority Level (Weight: 15.0%)")
   - Extract just the name before the colon ":"
   - Example: "Seniority Level (Weight: 15.0%): description" → criteria_name should be "Seniority Level"
2. "score" - number between 0-100
3. "evidence" - REQUIRED: specific evidence from the CV (quote or paraphrase). DO NOT leave empty!
4. "gap" - REQUIRED if score < 80: what's missing or below requirement. Leave empty string "" if score >= 80.

Return ONLY valid JSON with ALL fields populated. Evidence and gap fields are MANDATORY."""
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
            max_tokens=4000,  # Increased to allow for evidence/gap text
            generation_name="criteria_scoring_llm",
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        
        print(f"✓ Criteria scoring LLM call: {llm_duration:.2f}s")
        print(f"LLM Response (first 500 chars): {response_text[:500]}...")
        print(f"LLM Response length: {len(response_text)} chars")
        
        # Try to parse JSON (handle potential markdown wrapping and extra text)
        response_text_original = response_text
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if "```json" in response_text:
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        elif "```" in response_text:
            start_idx = response_text.find("```") + 3
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                response_text = response_text[start_idx:end_idx].strip()
        
        # Try to find JSON object boundaries
        if "{" in response_text and "}" in response_text:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
        
        response_text = response_text.strip()
        
        # Validate we have something to parse
        if not response_text:
            raise ValueError(f"Empty response after parsing. Original response: {response_text_original[:500]}")
        
        # Parse JSON with better error message
        try:
            scores_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing failed at position {e.pos}: {e.msg}\n"
            error_msg += f"Response text (first 1000 chars):\n{response_text[:1000]}\n"
            error_msg += f"Original response (first 500 chars):\n{response_text_original[:500]}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg) from e
        
        # Validate response structure
        if "criteria_scores" not in scores_data:
            raise ValueError(f"Missing 'criteria_scores' key in response. Keys found: {list(scores_data.keys())}")
        
        # Debug: Print first score to see structure
        if len(scores_data["criteria_scores"]) > 0:
            first_score = scores_data["criteria_scores"][0]
            print(f"DEBUG - First score structure: {json.dumps(first_score, indent=2)}")
            print(f"DEBUG - Keys in first score: {list(first_score.keys())}")
        
        # Create a mapping from criterion names (with or without weight) to actual criterion names
        criterion_name_map = {}
        for criterion in rubric.criteria:
            # Map the exact name
            criterion_name_map[criterion.name] = criterion.name
            # Map name with weight format (as shown in prompt)
            criterion_name_map[f"{criterion.name} (Weight: {criterion.weight:.1f}%)"] = criterion.name
        
        # Convert to CriterionScore list
        scores = []
        for s in scores_data["criteria_scores"]:
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
            
            # Debug: Log name normalization
            if raw_criteria_name != normalized_name:
                print(f"DEBUG: Normalized criteria name: '{raw_criteria_name}' -> '{normalized_name}'")
            
            score_obj = CriterionScore(
                criteria_name=normalized_name,  # Use normalized name
                score=float(s["score"]),
                evidence=s.get("evidence", "") or "",  # Ensure it's a string, not None
                gap=s.get("gap", "") or ""  # Ensure it's a string, not None
            )
            
            # Debug: Print if evidence/gap are empty
            if not score_obj.evidence and not score_obj.gap:
                print(f"WARNING: No evidence or gap for criterion: {score_obj.criteria_name}")
            
            scores.append(score_obj)
        
        print(f"✓ Scored {len(scores)} criteria via LLM")
        
        # Debug: Print summary of evidence/gap
        evidence_count = sum(1 for s in scores if s.evidence)
        gap_count = sum(1 for s in scores if s.gap)
        print(f"DEBUG - Scores with evidence: {evidence_count}/{len(scores)}")
        print(f"DEBUG - Scores with gap: {gap_count}/{len(scores)}")
        
        # LANGFUSE: Span updated/closed automatically
        
        return scores
        
    except Exception as e:
        print(f"ERROR in LLM call: {e}")
        # LANGFUSE: Error captured automatically by @observe
        raise


def generate_qualification_note(
    job_posting: str,
    cv_profile: str,
    session_id: str = None,
    model: str = None
) -> str:
    """
    Generate a comprehensive qualification note for a candidate.
    
    Args:
        job_posting: The job posting text
        cv_profile: The candidate's CV text
        session_id: Optional session ID for Langfuse tracking
        
    Returns:
        HTML-formatted qualification note
    """
    print("\n[LLM CALL via OpenRouter] Candidate Qualification Generation...")
    
    # Build structured context like the actual implementation
    base_context = "### INPUTS\n\n"
    
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

### TASK:
**You have received all the required information above.** The job posting and candidate résumé have been provided. 

DO NOT ask for documents. DO NOT wait for input. 

**GENERATE THE COMPLETE QUALIFICATION NOTE NOW** following the exact HTML structure specified in your instructions.

Start your response directly with:
<b>OVERALL ASSESSMENT: [Fit Level]</b>
"""
    
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
            compiled = langfuse_prompt.compile(
                job_posting=job_posting,
                cv_profile=cv_profile
            )
            
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
            langfuse_prompt=langfuse_prompt,
            session_id=session_id,
            model=model
        )
        # print(f"✓ Generated qualification note : {response_text[:200]}")
        
        print(f"✓ Generated qualification note ({len(response_text)} chars, LLM: {llm_duration:.2f}s)")
        
        return response_text
        
    except Exception as e:
        print(f"❌ Qualification generation failed: {e}")
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

