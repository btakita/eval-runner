.PHONY: build release test clippy check eval eval-smoke eval-regression eval-json install clean

CASES ?= ../../evals/agent-doc/diff-interpretation/cases.yaml
BASELINE ?= ../../evals/agent-doc/diff-interpretation/baselines/sonnet-2026-04-04.json
MODEL ?= sonnet
JUDGE_MODEL ?= sonnet
PARALLEL ?= 8
SYSTEM_PROMPT ?= ../../evals/agent-doc/diff-interpretation/system-prompt.md
SYSTEM_PROMPT_FLAG = $(if $(wildcard $(SYSTEM_PROMPT)),--system-prompt $(SYSTEM_PROMPT),)

build:
	cargo build

release:
	cargo build --release
	@mkdir -p .bin
	@ln -sf ../target/release/eval-runner .bin/eval-runner
	@echo "Installed .bin/eval-runner -> target/release/eval-runner"

test:
	cargo test

clippy:
	cargo clippy -- -D warnings

check: clippy test

install:
	cargo install --path .

eval:
	cargo run --release -- run $(CASES) --model $(MODEL) --judge-model $(JUDGE_MODEL) --parallel $(PARALLEL) $(SYSTEM_PROMPT_FLAG)

eval-smoke:
	cargo run --release -- run $(CASES) --model $(MODEL) --judge-model $(JUDGE_MODEL) --parallel $(PARALLEL) --tag smoke $(SYSTEM_PROMPT_FLAG)

eval-regression:
	cargo run --release -- run $(CASES) --model $(MODEL) --judge-model $(JUDGE_MODEL) --parallel $(PARALLEL) --baseline $(BASELINE) $(SYSTEM_PROMPT_FLAG)

eval-json:
	cargo run --release -- run $(CASES) --model $(MODEL) --judge-model $(JUDGE_MODEL) --parallel $(PARALLEL) --json $(SYSTEM_PROMPT_FLAG)

clean:
	cargo clean
	rm -f .bin/eval-runner
