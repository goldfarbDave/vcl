SHELL := /bin/bash
VENV := vclvenv
DATA_DIR := pytorch/data/
CIFAR_DATA := $(DATA_DIR)/cifar-10-batches-py
CIFAR_PKL := $(DATA_DIR)/cifar-10.pkl
CIFAR_TRANS := pytorch/data/cifar_transform.py
all: $(VENV)/pipreqsdone $(CIFAR_PKL)
	@bash --init-file <(echo " . ~/.bash_profile; . $(VENV)/bin/activate;")
	@echo "done"

$(VENV)/pipreqsdone: requirements.txt | $(VENV)
	source $(VENV)/bin/activate && \
	pip install wheel && \
	pip install -r requirements.txt
	touch $(VENV)/pipreqsdone

$(VENV):
	python3 -m venv $(VENV)

$(CIFAR_PKL): $(CIFAR_DATA) $(CIFAR_TRANS)
	python3 $(CIFAR_TRANS) -srcdir=$(CIFAR_DATA) -outpkl=$(CIFAR_PKL)

$(CIFAR_DATA):
	wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	tar xvf cifar-10-python.tar.gz
	mv cifar-10-batches-py/ $(DATA_DIR)
	rm cifar-10-python.tar.gz

.PHONY: cleancifar

cleancifar:
	rm -rf $(CIFAR_DATA)
	rm -rf $(CIFAR_PKL)

clean: cleancifar
	rm -rf $(VENV)
