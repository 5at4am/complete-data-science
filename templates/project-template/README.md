# Project Template

> **Every substantial project must follow this structure.**
> Copy this template for each new project.

---

## Project Structure

```
project-name/
├── README.md              # Project overview
├── requirements.txt       # Dependencies
├── data/                  # Data (raw, processed)
├── notebooks/             # Exploration and experiments
├── src/                   # Source code
│   ├── data/              # Data loading/processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions
│   ├── evaluation/        # Evaluation code
│   └── utils/             # Utilities
├── tests/                 # Tests
├── config/                # Configuration
└── models/                # Trained models
```

---

## README Template

```markdown
# [Project Name]

## Problem Definition
[What problem does this solve?]

## Business/Real-World Context
[Why does this matter?]

## Requirements
[What must the system do?]

## Constraints
[What limits exist? (data, compute, latency, cost)]

## Data Understanding
[Describe the data: source, size, features, target]

## Baseline
[What is the naive/trivial baseline?]

## Model Selection
[What models were considered? Why?]

## Experiment Design
[What experiments were run?]

## Evaluation
[How was the system evaluated? What metrics?]

## Error Analysis
[Where does it fail? Why?]

## Iteration
[How was the system improved?]

## Final Architecture
[Describe the final system]

## Implementation
[How to run it]

## Testing
[How is it tested?]

## Documentation
[Links to docs]

## Limitations
[What are the known limitations?]

## Future Improvements
[What could be improved?]
```

---

## Project Requirements Checklist

- [ ] Problem definition
- [ ] Business/real-world context
- [ ] Requirements
- [ ] Constraints
- [ ] Data understanding
- [ ] Baseline
- [ ] Model selection
- [ ] Experiment design
- [ ] Evaluation
- [ ] Error analysis
- [ ] Iteration
- [ ] Final architecture
- [ ] Implementation
- [ ] Testing
- [ ] Documentation
- [ ] Limitations
- [ ] Future improvements

---

## Project Levels

| Level | Description | Guidance |
|-------|-------------|----------|
| 1 | Guided | Everything specified |
| 2 | Partially guided | Dataset + objective, decisions left to learner |
| 3 | Open-ended | Only problem statement |
| 4 | Engineering challenge | Requirements + constraints + evaluation |
| 5 | Production capstone | Full system design |
