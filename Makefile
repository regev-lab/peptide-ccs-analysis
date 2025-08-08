#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python

ENV ?= ccs-analysis
PYTHON ?= 3.11
CONDA ?= conda

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: install
install:
	$(CONDA) run -n $(ENV) pip install -e .

## Install Python dev dependencies
.PHONY: install_dev
install_dev:
	$(CONDA) run -n $(ENV) pip install -e .[dev]


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(CONDA) run -n $(ENV) ruff format --check
	$(CONDA) run -n $(ENV) ruff check


## Format source code with ruff
.PHONY: format
format:
	$(CONDA) run -n $(ENV) ruff check --fix
	$(CONDA) run -n $(ENV) ruff format


## Run tests
.PHONY: test
test:
	$(CONDA) run -n $(ENV) pytest -q tests


## Set up Python interpreter environment
.PHONY: env
env:
	$(CONDA) create -y -n $(ENV) python=$(PYTHON)
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
