SHELL := /bin/bash
VENV := vclvenv
CIFAR_DATA_DIR := pytorch/data/
CIFAR_DATA := $(CIFAR_DATA_DIR)/cifar-10-batches-py
CIFAR_PKL := $(CIFAR_DATA_DIR)/cifar-10.pkl
CIFAR_TRANS := pytorch/data/cifar_transform.py
NOTMNIST_DATA_DIR := tensorflow/dgm/data/
NOTMNIST_DATA := $(NOTMNIST_DATA_DIR)/notMNIST_small.mat

all: $(VENV)/pipreqsdone $(CIFAR_PKL) $(NOTMNIST_DATA)
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
	mv cifar-10-batches-py/ $(CIFAR_DATA_DIR)
	rm cifar-10-python.tar.gz

$(NOTMNIST_DATA): | $(NOTMNIST_DATA_DIR)
	wget "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat" -O $(NOTMNIST_DATA)

$(NOTMNIST_DATA_DIR):
	mkdir -p $(NOTMNIST_DATA_DIR)


.PHONY: cleancifar cleannotmnist

cleancifar:
	rm -rf $(CIFAR_DATA)
	rm -rf $(CIFAR_PKL)

cleannotmnist:
	rm -rf $(NOTMNIST_DATA)

clean: cleancifar
	rm -rf $(VENV)
