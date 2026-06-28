# Usage examples:
#   make train component 3000 10
#   make train connected 10000 15 EPOCHS=75
#   make train trefoil 2000 8 PYTHON=.\.venv\Scripts\python.exe
#
# Positional arguments for train are: <type> <num_samples> <dimension>.

PYTHON ?= python
EPOCHS ?= 50

TYPE := $(word 2,$(MAKECMDGOALS))
SAMPLES := $(word 3,$(MAKECMDGOALS))
DIMENSION := $(word 4,$(MAKECMDGOALS))

ifeq ($(filter train,$(MAKECMDGOALS)),train)
ifeq ($(TYPE),)
$(error Usage: make train <type> <num_samples> <dimension> [EPOCHS=<epochs>])
endif
ifeq ($(SAMPLES),)
$(error Usage: make train <type> <num_samples> <dimension> [EPOCHS=<epochs>])
endif
ifeq ($(DIMENSION),)
$(error Usage: make train <type> <num_samples> <dimension> [EPOCHS=<epochs>])
endif
endif

.PHONY: help train

help:
	@echo "Usage:"
	@echo "  make train <type> <num_samples> <dimension>"
	@echo "  make train <type> <num_samples> <dimension> EPOCHS=<epochs>"
	@echo ""
	@echo "Examples:"
	@echo "  make train component 3000 10"
	@echo "  make train connected 10000 15 EPOCHS=75"

train:
	$(PYTHON) data_gen.py $(TYPE) $(SAMPLES) $(DIMENSION)
	$(PYTHON) cnn_train.py $(TYPE) $(EPOCHS) $(DIMENSION)

%:
	@:
