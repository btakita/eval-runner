.PHONY: build release test clippy check install clean

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

clean:
	cargo clean
	rm -f .bin/eval-runner
