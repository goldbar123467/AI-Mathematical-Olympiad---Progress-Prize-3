# AIMO Progress Prize 3

Solution for [Kaggle AI Mathematical Olympiad - Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Local evaluation
python main.py --test data/reference.csv

# Training (requires GPU)
python train.py
```

## Project Structure

```
├── main.py              # Kaggle submission entry point
├── train.py             # Fine-tuning script
├── src/
│   ├── models/          # Model wrappers (vLLM, HuggingFace)
│   ├── solvers/         # Solving strategies (CoT, ensemble)
│   ├── parsers/         # LaTeX parsing
│   └── utils/           # Helpers
├── data/                # Competition data
├── kaggle_evaluation/   # Official eval API
└── notebooks/           # Experiments
```

## Competition Details

- **Prize**: $2.2M + $5M grand prize
- **Deadline**: April 2026
- **Format**: 110 olympiad-level math problems, 5-digit integer answers
- **Hardware**: H100 GPUs on Kaggle

See `COMPETITION_PLAN.md` for full strategy.
