.PHONY: setup reproduce test demo originals

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements-lock.txt

reproduce:
	.venv/bin/python download_data.py
	.venv/bin/python run_experiments.py

test:
	.venv/bin/python -m unittest discover -s tests -v

demo:
	.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

originals:
	.venv/bin/python train.py
	.venv/bin/python generate_visuals.py
