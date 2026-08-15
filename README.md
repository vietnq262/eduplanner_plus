# EduPlanner+: LLM-based Multi-Agent System for Personalized Lesson Plan Generation in Multi-Domain Adaptive Learning

## What this is

EduPlanner+ is an LLM-based multi-agent framework that generates personalized lesson plans for adaptive learning across multiple domains. It extends the prior EduPlanner framework by integrating five specialized agents to jointly handle dynamic student ability modeling, self-reflective lesson refinement, and weak-ability-targeted question selection—evaluated across algebra, math word problems (GSM8K), databases, and operating systems domains.

## Stack

- **Language(s):** Python
- **Framework / runtime:** OpenAI API (GPT-4o, GPT-4o-mini) via OpenRouter
- **Notable libraries:** pandas (dataset handling), tqdm (progress tracking), httpx (HTTP), OpenAI Python client

## How it's organized

```
codes/
  oaeo8.py              Main orchestrator: Optimizer Agent Expert Optimization pipeline
  eaee3.py              Evaluator Agent Expert Evaluation (multi-expert lesson scoring)
  aaea.py               Analyst Agent Expert Analysis (error-prone point augmentation)
  reflection.py         Reflection Agent (self-critique-based lesson refinement)
  util3.py              Utilities: Dynamic Skill Tree, student ability profiling
  
  pre_prompt_gsm8k.py   Configuration: hyperparameters, prompts, API setup (example for GSM8K)
  pre_prompt_algebra.py Domain-specific configs: Algebra
  pre_prompt_sql.py     Domain-specific configs: SQL/Database
  pre_prompt_os.py      Domain-specific configs: Operating Systems
  
  data_processing.py    Dataset utilities
  CIDPP2.py             CIDPP evaluation (final multi-criteria assessment)
  aqs.py                Adaptive Question Selector (weak-ability targeting)

datasets/
  algebra222.csv        Algebra benchmark dataset
  gsm8k_train.csv       Math word problems training data
  gsm8k_test.csv        Math word problems test data
  questions_sql.csv     Database system questions
  questions_os.csv      Operating system questions
```

## How it fits together

The pipeline starts with an initial lesson plan and enters an optimization loop (N rounds). Each round follows this sequence:

1. **EAEE (Evaluator Agent Expert Evaluation):** Scores the current lesson plan against T selected questions, considering the student's current ability persona.

2. **Reflection Agent:** Critiques identified weaknesses and refines the lesson plan based on EAEE feedback.

3. **AQS (Adaptive Question Selector):** Dynamically selects T questions that specifically target the student's weak abilities (rather than random sampling).

4. **AAEA (Analyst Agent Expert Analysis):** Augments the lesson plan with common mistakes and their probabilities to highlight error-prone points.

5. **Dynamic Skill Tree:** Evolves the student ability profile based on EAEE scores:
   - **Upgrade:** If score > high_threshold (80), upgrade the weakest ability by +1
   - **Downgrade:** If score < low_threshold (40), downgrade the strongest ability by -1
   

6. **CIDPP Evaluation:** Final assessment scores the best lesson plans on six criteria:
   - **Clarity:** Clear and understandable explanations
   - **Depth:** Deep conceptual understanding with connections
   - **Integrity:** Complementary Part 1 (knowledge) and Part 2 (exercises)
   - **Practicality:** Real-world scenario applications
   - **Pertinence:** Tailored to student's ability profile
   - **Constraints:** Adherence to structural requirements

Each lesson plan consists of exactly **two parts**:
- **Part 1:** Knowledge Point Explanation (concepts, WHY they work, conceptual connections)
- **Part 2:** Exercise Explanation (real-world problems with step-by-step solutions and common mistakes)

## How to run it

### Prerequisites

```bash
pip install openai pandas tqdm httpx
```

### Configuration

Edit the domain-specific configuration file (e.g., `codes/pre_prompt_gsm8k.py`):

```python
# Hyperparameters
N = 20  # optimization rounds
T = 8   # test questions per evaluation
M = 5   # lesson plan identifier
K = 3   # lesson plans generated per optimization round
P = 3   # max lesson plans kept in optimization prompt

# API Setup
API_KEY = "your-openrouter-api-key"
BASE_URL = "https://openrouter.ai/api/v1"

# Dataset and ability tree paths
DATASET_PATH = os.path.join(ROOT_DIR, "datasets/gsm8k_train.csv")
ABILITY_TREE_PATH = os.path.join(ROOT_DIR, "persona/ability_tree_gsm8k.json")
```

### Running the optimizer

For GSM8K (math word problems):
```bash
python codes/oaeo8.py
```

For other domains (Algebra, SQL, Operating Systems):
- Modify the imports in `oaeo8.py` to use the appropriate `pre_prompt_*.py` file
- Adjust `DATASET_PATH` and `ABILITY_TREE_PATH` accordingly

### Output

Results are saved to `result/result_YYYYMMDD_HHMMSS.json` containing:
- Timestamp and hyperparameters
- Final student ability profiles
- Optimized lesson plans with EAEE scores
- CIDPP evaluation scores for each lesson plan

## Key Components

### 1. Evaluator Agent Expert Evaluation (EAEE)
- **File:** `codes/eaee3.py`
- Evaluates lesson plan quality by testing against T questions with the current student persona
- Returns: average score, identified advantages, and disadvantages
- Uses multi-expert evaluation: each question is scored independently by GPT-4o-mini

### 2. Reflection Agent
- **File:** `codes/reflection.py`
- Self-critique mechanism: reviews EAEE feedback and identifies 3 weakest parts
- Refines only weak sections while preserving strong parts
- Re-evaluates refined plan; only keeps improvement if score increases

### 3. Analyst Agent Expert Analysis (AAEA)
- **File:** `codes/aaea.py`
- Augments lesson plans with common mistakes database
- Identifies error-prone points and estimates mistake probabilities
- Integrates common mistakes into exercise explanations

### 4. Adaptive Question Selector (AQS)
- **File:** `codes/aqs.py`
- Dynamically selects T questions targeting student weak abilities
- Improves evaluation validity by focusing on areas of difficulty

### 5. Dynamic Skill Tree
- **File:** `codes/util3.py`
- Tracks 5 core student abilities (varies by domain)
- Updates ability levels based on EAEE scores each round
- Generates persona descriptions for LLM context

## Benchmark Datasets

- **Algebra:** 222 algebra problems
- **GSM8K:** 1,300 training + 1,319 test grade school math word problems
- **SQL/Database:** Database system questions
- **Operating Systems:** OS fundamentals questions

## Publication Context

This is the reference implementation for the paper:

> **EduPlanner+: Dynamic Student Ability Evolution and Self-Reflective Lesson Refinement for Multi-Domain Adaptive Learning**
>
> An extension of EduPlanner that adds:
> - Dynamic student ability evolution via skill tree updates
> - Self-reflective lesson refinement via Reflection Agent
> - Weak-ability-targeted question selection via Adaptive Question Selector
> - Multi-expert lesson evaluation via CIDPP criteria

## Citation

If you use EduPlanner+ in your research, please cite:

```bibtex
@software{eduplannerplus2024,
  title={EduPlanner+: LLM-based Multi-Agent System for Personalized Lesson Plan Generation in Multi-Domain Adaptive Learning},
  author={Nguyen, Viet Quoc},
  year={2026},
  url={https://github.com/vietnq262/eduplannerplus}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainers.
