# Detailed Issue Analysis: `Makefile`

**File:** [`Makefile`](file:///home/nyahyaou/Desktop/RAG/Makefile)  
**Severity:** 🟢 **MEDIUM** (Broken flags in `run`, invalid syntax in `debug`, undeclared dependency in `clean`)

---

## 1. Overview of Problems

| Bug # | Type | Location | Impact |
|---|---|---|---|
| **8.1** | Broken Short Flags in `run` Target | Lines 31–32 | Passes `-d` and `-s` to Python Fire, which expects `--dataset_path` and `--student_search_results_path`. Causes `search_dataset` to ignore the code dataset! |
| **8.2** | Undeclared Dependency in `clean` | Line 39 | `$(RUN) pyclean . -d` fails if `pyclean` is not installed. |
| **8.3** | Invalid `pdb` Syntax in `debug` | Line 36 | `$(RUN) -m pdb -m src` is invalid pdb syntax. |
| **8.4** | Missing `lint-strict` Target | Common Instructions V.2 | Subject recommends including `lint-strict` (`flake8 . && mypy . --strict`). |

---

## 2. In-Depth Root Cause Analysis

### Issue 8.1: The `-d` and `-s` Flag Bug in `make run`
Look at lines 31–32 of `Makefile`:
```makefile
@$(RUN) -m src search_dataset -d $(DATASET_CODE_PATH)  -k $(TOP_K)
@$(RUN) -m src evaluate -s $(STUDENT_SEARCH_CODE_RESULT) -d $(TRUTH_DATASET_CODE_PATH)
```
In Python Fire, CLI arguments match parameter names:
```python
def search_dataset(self, dataset_path=SEARCH_DATASET_PATH, save_directory=SEARCH_SAVE_DIRECTORY, k=TOP_K):
```
Because the parameter is named `dataset_path`, Python Fire expects `--dataset_path`.
* When you pass `-d`, Fire creates an unexpected keyword argument `d=...`.
* The parameter `dataset_path` remains at its **default value** (`dataset_docs_public.json`)!
* **Result**: `make run` silently re-evaluates the docs dataset instead of evaluating the code dataset!

#### Fix:
Use the full flag names:
```makefile
@$(RUN) -m src search_dataset --dataset_path $(DATASET_CODE_PATH) --k $(TOP_K)
@$(RUN) -m src evaluate --student_search_results_path $(STUDENT_SEARCH_CODE_RESULT) --dataset_path $(TRUTH_DATASET_CODE_PATH)
```

---

### Issue 8.2: Broken `make clean`
```makefile
clean:
	$(RUN) pyclean . -d
	rm -rf .mypy_cache
```
`pyclean` is not included in `pyproject.toml`. If a peer reviewer or grading machine does not have `pyclean`, running `make clean` halts with an error.

#### Fix:
Use standard POSIX `find` and `rm`:
```makefile
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache data/cache
```

---

### Issue 8.3: Invalid `pdb` Invocation
```makefile
debug:
	$(RUN) -m pdb -m src
```
`pdb` takes a file path, not a module `-m` flag. This will fail with `Error: -m does not exist`.

#### Fix:
```makefile
debug:
	$(RUN) python -m pdb src/__main__.py
```

---

## 3. Corrected `Makefile`

```makefile
UV = uv
RUN = $(UV) run

DATASET_CODE_PATH = data/datasets/public/UnansweredQuestions/dataset_code_public.json
TRUTH_DATASET_CODE_PATH = data/datasets/public/AnsweredQuestions/dataset_code_public.json
STUDENT_SEARCH_CODE_RESULT = data/output/search_results/dataset_code_public.json

TOP_K = 10

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	$(UV) sync

run:
	$(RUN) python -m src index
	@echo ""
	@echo "====================================================="
	@echo "                EVALUATE DOCS DATASET"
	@echo "====================================================="
	@$(RUN) python -m src search_dataset --k $(TOP_K)
	@$(RUN) python -m src evaluate
	@echo ""
	@echo "====================================================="
	@echo "                EVALUATE CODE DATASET"
	@echo "====================================================="
	@$(RUN) python -m src search_dataset --dataset_path $(DATASET_CODE_PATH) --k $(TOP_K)
	@$(RUN) python -m src evaluate --student_search_results_path $(STUDENT_SEARCH_CODE_RESULT) --dataset_path $(TRUTH_DATASET_CODE_PATH)

debug:
	$(RUN) python -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache

lint:
	$(RUN) flake8 .
	$(RUN) mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(RUN) flake8 .
	$(RUN) mypy . --strict
```

