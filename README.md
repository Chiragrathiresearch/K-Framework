## Citation
If you use this work, please cite the Zenodo record:
DOI: 10.5281/zenodo.18342502

---

# K-Framework 

**K-Framework** is an uncertainty-aware, research-grade framework for satellite
state estimation, propagation, and afterlife analysis using public orbital data.

The framework is designed as a modular scientific system rather than an
application, emphasizing epistemic uncertainty, layered physics modeling,
and architectural separation of concerns.

---

## Motivation

Modern satellite analysis often treats state estimates as deterministic outputs.
K-Framework is built on the premise that **every observation carries uncertainty**
and that this uncertainty should be explicitly modeled, propagated, and reasoned
about across the analysis pipeline.

This project is intended for research exploration, inspection, and extension.

---

## Framework Structure

The framework is organized into conceptual layers:

- **K19 — Uncertainty Layer**  
  Epistemic uncertainty representation and propagation logic.

- **K20 — Physics Layer**  
  Physics-based satellite state modeling and propagation.

- **K21 — Architecture Layer**  
  Filtering logic, memory abstraction, and system coordination.

Supporting modules handle observation ingestion, workflow execution, and
satellite–data interfacing.

---

## Repository Contents

- `k19_uncertainty.py` — Uncertainty modeling
- `k20_physics.py` — Physics-based state representation
- `k21_memory.py` — Architectural memory abstraction
- `observation_ingestion.py` — Observation and data intake
- `satellite_bridge.py` — Interface to orbital data
- `workflow.py` — End-to-end research workflow
- `run_research.py` — Execution entry point
- `settings.py` — Configuration
- `requirements.txt` — Dependency specification

---
