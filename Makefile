.PHONY: train test parity replay

train:
	python -m ridcheck train

test:
	python -m pytest -q

parity:
	gcc -O2 -std=c11 -Wall -Wextra -o /tmp/ridcheck_parity tests/parity_main.c firmware/src/ridcheck_infer.c -lm
	/tmp/ridcheck_parity

replay:
	python -m ridcheck replay examples/sample_scene.jsonl
