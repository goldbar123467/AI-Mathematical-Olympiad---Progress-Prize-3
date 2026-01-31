# AIMO Progress Prize 3 - Competition Plan

## Competition Overview

**Prize Pool:** $2.2M main + $110K bonus prizes + $5M grand prize (first IMO gold-capable model)
**Deadline:** April 2026 (public phase)
**Showcase:** AI Day at 2026 IMO in Shanghai, China

### Key Facts
- **110 original problems** in algebra, combinatorics, geometry, number theory
- Difficulty: National Olympiad → IMO standard
- **5-digit answers** (no guessing possible)
- **9-hour runtime** per submission
- **H100 GPUs** available for training and inference
- **Open-source requirement** for prize eligibility

---

## Data Format

### Input (`test.csv`)
```csv
"id","problem"
"000aaa","What is $1-1$?"
```

### Output (`submission.csv`)
```csv
id,answer
000aaa,0
```

- Answers are **integers** (5-digit capable range: 0-99999)
- Problems contain **LaTeX math** notation
- Problems are shuffled during evaluation

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KAGGLE EVALUATION                        │
├─────────────────────────────────────────────────────────────┤
│  Gateway (aimo_3_gateway.py)                                │
│  ├─ Reads test.csv                                          │
│  ├─ Shuffles problems (public) / fixed order (private)      │
│  ├─ Yields one problem at a time                            │
│  └─ 9-hour timeout                                          │
├─────────────────────────────────────────────────────────────┤
│  Your Inference Server                                      │
│  ├─ Receives: {"id": "xxx", "problem": "LaTeX text"}        │
│  ├─ Returns: {"id": "xxx", "answer": integer}               │
│  └─ Must handle all problem types                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Environment Setup

### 1.1 Project Structure
```
aimo-progress-prize-3/
├── data/
│   ├── reference.csv          # 10 example problems with answers
│   ├── test.csv               # Dummy test set
│   └── AIMO3_Reference_Problems.pdf
├── src/
│   ├── models/                # Model wrappers
│   ├── solvers/               # Math solving strategies
│   ├── parsers/               # LaTeX parsing
│   └── utils/                 # Helpers
├── notebooks/                 # Experimentation
├── submissions/               # Submission versions
├── kaggle_evaluation/         # Official eval code
├── requirements.txt
├── Dockerfile
└── main.py                    # Entry point
```

### 1.2 Dependencies
```txt
torch>=2.0
transformers>=4.40
vllm>=0.4.0
polars
sympy
latex2sympy2
grpcio
protobuf
```

---

## Phase 2: Model Selection

### Recommended Models (H100 optimized)

| Model | Parameters | Strength | Use Case |
|-------|------------|----------|----------|
| **Qwen2.5-Math-72B-Instruct** | 72B | Best open-source math | Primary solver |
| **DeepSeek-Math-7B-RL** | 7B | Efficient reasoning | Fast fallback |
| **Numina-Math-7B** | 7B | AIMO1 winner base | Ensemble member |
| **Mathstral-7B** | 7B | Mistral math fine-tune | Ensemble member |

### Strategy: Multi-Model Ensemble
1. Primary: Qwen2.5-Math-72B with chain-of-thought
2. Secondary: DeepSeek-Math-7B for verification
3. Fallback: Symbolic solver (SymPy) for tractable problems

---

## Phase 3: Solving Pipeline

### 3.1 Problem Classification
```python
def classify_problem(problem: str) -> str:
    """Classify into: algebra, combinatorics, geometry, number_theory"""
    keywords = {
        'algebra': ['polynomial', 'equation', 'root', 'function'],
        'combinatorics': ['count', 'permutation', 'sequence', 'ways'],
        'geometry': ['triangle', 'circle', 'angle', 'point'],
        'number_theory': ['divisor', 'prime', 'modulo', 'integer']
    }
    # Use LLM classification as backup
```

### 3.2 Solving Strategies

#### A. Chain-of-Thought (CoT)
```python
SYSTEM_PROMPT = """You are an expert mathematician solving olympiad problems.
Think step by step. Show all work. Give final answer as integer."""

def solve_cot(problem: str, model) -> int:
    response = model.generate(
        system=SYSTEM_PROMPT,
        user=problem,
        temperature=0.1,
        max_tokens=8192
    )
    return extract_answer(response)
```

#### B. Self-Consistency (majority voting)
```python
def solve_self_consistent(problem: str, model, n_samples=8) -> int:
    answers = []
    for _ in range(n_samples):
        answer = solve_cot(problem, model, temperature=0.7)
        answers.append(answer)
    return majority_vote(answers)
```

#### C. Program-of-Thought (PoT)
```python
def solve_pot(problem: str, model) -> int:
    code = model.generate(
        system="Write Python code to solve this math problem.",
        user=problem
    )
    result = safe_exec(code)  # Sandboxed execution
    return int(result)
```

#### D. Tool-Augmented Reasoning
```python
def solve_with_tools(problem: str, model) -> int:
    tools = [
        sympy_solver,      # Symbolic math
        wolfram_api,       # Computation
        geometry_solver,   # Geometric constructions
    ]
    return model.generate_with_tools(problem, tools)
```

### 3.3 Answer Extraction
```python
import re

def extract_answer(response: str) -> int:
    patterns = [
        r'answer[:\s]+(\d+)',
        r'result[:\s]+(\d+)',
        r'\*\*(\d+)\*\*',
        r'\\boxed\{(\d+)\}',
        r'= (\d+)$'
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1)) % 100000  # 5-digit constraint
    return 0  # Fallback
```

---

## Phase 4: Training Strategy

### 4.1 Fine-tuning Data Sources
1. **MATH dataset** (12K competition problems)
2. **GSM8K** (8.5K grade school math)
3. **Numina-Math-CoT** (860K synthetic solutions)
4. **AIMO reference problems** (10 problems for validation)
5. **AoPS problems** (Art of Problem Solving archives)

### 4.2 Fine-tuning Approach
```python
# LoRA fine-tuning for efficiency on H100
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Train on chain-of-thought solutions
# Reward model: correct answer = +1, incorrect = -1
```

### 4.3 Reinforcement Learning (optional)
- RLHF with outcome-based rewards
- Process reward models (PRM) for step verification

---

## Phase 5: Inference Optimization

### 5.1 vLLM Configuration
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-Math-72B-Instruct",
    tensor_parallel_size=4,  # H100 x4
    gpu_memory_utilization=0.95,
    max_model_len=8192
)

sampling = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=4096,
    stop=["Answer:", "\\boxed{"]
)
```

### 5.2 Batching Strategy
- Problems arrive one at a time (can't batch)
- Use speculative decoding for speedup
- Cache common sub-computations

---

## Phase 6: Submission Template

```python
# main.py - Kaggle submission entry point
import polars as pl
from kaggle_evaluation.core.templates import InferenceServer

class AIMO3Solver(InferenceServer):
    def __init__(self):
        self.model = load_model()

    def predict(self, id: str, problem: str) -> dict:
        # 1. Classify problem
        category = classify_problem(problem)

        # 2. Select solver strategy
        if category == 'algebra':
            answer = self.solve_algebra(problem)
        elif category == 'geometry':
            answer = self.solve_geometry(problem)
        else:
            answer = self.solve_general(problem)

        # 3. Verify with secondary model
        answer = self.verify_answer(problem, answer)

        return {'id': id, 'answer': int(answer) % 100000}

if __name__ == '__main__':
    server = AIMO3Solver()
    server.run()
```

---

## Phase 7: Evaluation & Iteration

### 7.1 Local Validation
```bash
# Test with reference problems
python evaluate_local.py --input data/reference.csv
```

### 7.2 Metrics
- **Accuracy**: % of correct answers
- **Penalized accuracy**: Competition scoring (details TBD)
- **Solve time**: Average per problem (target < 5 min)

### 7.3 Ablation Studies
1. Model size: 7B vs 70B
2. Prompting: CoT vs PoT vs hybrid
3. Sampling: greedy vs self-consistency
4. Tools: with vs without symbolic math

---

## Timeline & Milestones

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | Environment, baseline model running |
| 2-3 | Baseline | Simple CoT solver, ~30% accuracy |
| 4-5 | Ensemble | Multi-model voting, ~50% accuracy |
| 6-7 | Fine-tune | Custom training on math data |
| 8-9 | Optimize | Inference speed, consistency |
| 10+ | Iterate | Based on leaderboard feedback |

---

## Quick Start Commands

```bash
# 1. Setup environment
cd aimo-progress-prize-3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download models (on Kaggle with H100)
huggingface-cli download Qwen/Qwen2.5-Math-72B-Instruct

# 3. Test locally
python main.py --test data/reference.csv

# 4. Create Kaggle submission
kaggle kernels push -p submissions/v1/
```

---

## Resources

- [Competition Page](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO Prize Official](https://aimoprize.com)
- [Submission Demo Notebook](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo)
- [Qwen2.5-Math Paper](https://arxiv.org/abs/2409.12122)
- [AIMO1 Winner (Numina) Write-up](https://projectnumina.ai/)

---

## Reference Problems Summary

The 10 reference problems cover:
1. **Geometry** - Triangle with circumcircle/incircle (answer: 336)
2. **Number Theory** - Floor function summation (answer: 32951)
3. **Combinatorics** - Tournament scoring orderings (answer: 21818)
4. **Number Theory** - Base conversion game (answer: 32193)
5. **Geometry** - Fibonacci triangle with circles (answer: 57447)
6. **Number Theory** - n-Norwegian numbers (answer: 8687)
7. **Algebra** - Age and sweets puzzle (answer: 50)
8. **Functional Equations** - f(m)+f(n)=f(m+n+mn) (answer: 580)
9. **Combinatorics** - Rectangle tiling (answer: 520)
10. **Algebra** - Shifty functions in Z→Z (answer: 160)
