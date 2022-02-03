SHELL := /bin/bash
VENV := vclvenv
all: $(VENV)/pipreqsdone
	@bash --init-file <(echo " . ~/.bash_profile; . $(VENV)/bin/activate;")
	@echo "done"

$(VENV)/pipreqsdone: requirements.txt | $(VENV)
	source $(VENV)/bin/activate && \
	pip install wheel && \
	pip install -r requirements.txt
	touch $(VENV)/pipreqsdone
$(VENV):
	python3 -m venv $(VENV)
clean:
	rm -rf $(VENV)
