.PHONY: help install test lint format run clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code"
	@echo "  make run         - Run main script"
	@echo "  make clean       - Clean generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ main.py --max-line-length=127
	black --check src/ tests/ main.py
	isort --check-only src/ tests/ main.py

format:
	black src/ tests/ main.py
	isort src/ tests/ main.py

run:
	python main.py

run-compare:
	python main.py --compare

run-config:
	python main.py --config configs/default_config.yaml

clean:
	rm -rf outputs/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete

docker-build:
	docker build -t isolation-forest-demo .

docker-run:
	docker run --rm -v $(PWD)/outputs:/app/outputs isolation-forest-demo
