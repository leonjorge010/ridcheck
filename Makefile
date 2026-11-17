.PHONY: train test parity replay

train:
	python -m spoofdet train

test:
	python -m pytest -q

parity:
	gcc -O2 -std=c11 -Wall -Wextra -o /tmp/spoofdet_parity tests/parity_main.c firmware/src/spoofdet_infer.c -lm
	/tmp/spoofdet_parity

replay:
	python -m spoofdet replay examples/sample_scene.jsonl
