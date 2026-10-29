.PHONY: install install-requirements install-api test run benchmark api docker-build clean

PYTHON ?= python3
PYTHONPATH_ENV = PYTHONPATH=src:.

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-requirements:
	$(PYTHON) -m pip install -r requirements.txt

install-api:
	$(PYTHON) -m pip install -e ".[api]"

test:
	$(PYTHONPATH_ENV) $(PYTHON) -B -m unittest discover -s tests -v

run:
	$(PYTHONPATH_ENV) $(PYTHON) -B scripts/run_video.py --config config/default.yaml --max-frames 45

api:
	$(PYTHONPATH_ENV) $(PYTHON) -B scripts/run_server.py --config config/default.yaml

benchmark:
	$(PYTHONPATH_ENV) $(PYTHON) -B scripts/benchmark.py

docker-build:
	docker build -t cctv-activity-system:1.0.0 .

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf outputs/*.avi outputs/*.jsonl
