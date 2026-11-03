UV = uv
RUN = $(UV) run

DATASET_CODE_PATH=data/datasets/public/UnansweredQuestions/dataset_code_public.json
TRUTH_DATASET_CODE_PATH=data/datasets/public/AnsweredQuestions/dataset_code_public.json
STUDENT_SEARCH_CODE_RESULT=data/output/search_results/dataset_code_public.json


TOP_K = 10


install:
	pip install uv
	$(UV) sync

run:
	$(RUN) -m src index
	@echo
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "                EVALUATE DOCS DATASET"
	@echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

	@$(RUN) -m src search_dataset -k $(TOP_K)
	@$(RUN) -m src evaluate
	@echo
	@echo
	@echo
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "                EVALUATE CODE DATASET"
	@echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
	@$(RUN) -m src search_dataset --dataset_path $(DATASET_CODE_PATH) -k $(TOP_K)
	@$(RUN) -m src evaluate --student_search_results_path $(STUDENT_SEARCH_CODE_RESULT) --dataset_path $(TRUTH_DATASET_CODE_PATH)


debug:
	$(RUN) python -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache


lint:
	$(RUN) flake8 .
	$(RUN) mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs


