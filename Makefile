# genos — Bare-Metal LLM Operating System
# Build and run targets for UEFI development

CARGO = cargo
TARGET = x86_64-unknown-uefi
BUILD_DIR = target/$(TARGET)/release
EFI_BINARY = $(BUILD_DIR)/genos-boot.efi
ESP_DIR = esp
ESP_IMG = esp.img
# ESP image size: 6G — enough for the 2.8G Gemma 4 GGUF + system files
# QEMU fat:rw has a 2GB-per-file limit; a raw FAT32 image has no such limit.

# OVMF firmware path — auto-detected for macOS (Intel + Apple Silicon) and Linux.
# On Apple Silicon (M1/M2/M3): brew install qemu installs OVMF at:
#   /opt/homebrew/share/qemu/edk2-x86_64-code.fd
# On Intel Mac (Homebrew): /usr/local/share/qemu/edk2-x86_64-code.fd
# On Linux: /usr/share/OVMF/OVMF_CODE.fd
OVMF_CODE ?= $(shell \
	if [ -f /opt/homebrew/share/qemu/edk2-x86_64-code.fd ]; then \
		echo "/opt/homebrew/share/qemu/edk2-x86_64-code.fd"; \
	elif [ -f /usr/local/share/qemu/edk2-x86_64-code.fd ]; then \
		echo "/usr/local/share/qemu/edk2-x86_64-code.fd"; \
	elif [ -f /usr/share/OVMF/OVMF_CODE.fd ]; then \
		echo "/usr/share/OVMF/OVMF_CODE.fd"; \
	elif [ -f /usr/share/edk2/x64/OVMF_CODE.fd ]; then \
		echo "/usr/share/edk2/x64/OVMF_CODE.fd"; \
	else \
		echo "OVMF_NOT_FOUND"; \
	fi)

# Apple Silicon note:
# qemu-system-x86_64 on M1/M2/M3 runs x86_64 via TCG software emulation.
# This is slower than native but fully functional for UEFI development and testing.
# Install with: brew install qemu  (brings OVMF firmware automatically)

.PHONY: build build-debug esp esp-img qemu qemu-debug qemu-net qemu-nographic clean help setup-model

# build-std flags: required to compile core/alloc from source for the UEFI target.
# These are passed explicitly here (not in .cargo/config.toml) so they don't
# affect other sub-crates in the repo (e.g., the std-based tests crate).
BUILD_STD = -Z build-std=core,compiler_builtins,alloc -Z build-std-features=compiler-builtins-mem

## Build in release mode (default — much faster in QEMU emulation)
build:
	$(CARGO) build --target $(TARGET) -p genos-boot --release $(BUILD_STD)

## Build debug binary (slow, for development only)
build-debug:
	$(CARGO) build --target $(TARGET) -p genos-boot $(BUILD_STD)

## Create the ESP (EFI System Partition) directory structure
esp: build
	@mkdir -p $(ESP_DIR)/EFI/BOOT
	@mkdir -p $(ESP_DIR)/models
	@mkdir -p $(ESP_DIR)/system
	@mkdir -p $(ESP_DIR)/palace/wings/general/halls/facts
	@mkdir -p $(ESP_DIR)/palace/wings/general/halls/events
	@mkdir -p $(ESP_DIR)/palace/wings/general/halls/discoveries
	@mkdir -p $(ESP_DIR)/palace/wings/general/halls/preferences
	@mkdir -p $(ESP_DIR)/palace/wings/general/halls/advice
	@mkdir -p $(ESP_DIR)/palace/sessions
	@cp $(EFI_BINARY) $(ESP_DIR)/EFI/BOOT/BOOTX64.EFI
	@if [ -f resources/identity.txt ] && [ ! -f $(ESP_DIR)/palace/identity.txt ]; then \
		cp resources/identity.txt $(ESP_DIR)/palace/identity.txt; \
	fi
	@if [ -f resources/config.toml ] && [ ! -f $(ESP_DIR)/system/config.toml ]; then \
		cp resources/config.toml $(ESP_DIR)/system/config.toml; \
	fi
	@if [ ! -f $(ESP_DIR)/palace/facts.kv ]; then \
		cp resources/facts.kv $(ESP_DIR)/palace/facts.kv 2>/dev/null || \
		echo "os_version = genos v0.1.0" > $(ESP_DIR)/palace/facts.kv; \
	fi
	@echo "ESP created at $(ESP_DIR)/"
	@echo ""
	@echo "Model files in $(ESP_DIR)/models/:"
	@if [ -f $(ESP_DIR)/models/gemma4-e2b.gguf ]; then \
		echo "  ✓ gemma4-e2b.gguf  (Gemma 4 E2B — primary)"; \
	else \
		echo "  ✗ gemma4-e2b.gguf  (missing — download from huggingface.co/unsloth/gemma-4-E2B-it-GGUF)"; \
	fi
	@if [ -f $(ESP_DIR)/models/tokenizer.json ]; then \
		echo "  ✓ tokenizer.json   (Gemma 4 tokenizer)"; \
	else \
		echo "  ✗ tokenizer.json   (missing — download from huggingface.co/google/gemma-4-E2B-it)"; \
	fi
	@if [ -f $(ESP_DIR)/models/stories15m.bin ]; then \
		echo "  ✓ stories15m.bin   (fallback model)"; \
	fi

## Build a raw FAT32 disk image from the ESP directory.
## Required because QEMU fat:rw has a 2GB per-file limit (breaks Gemma 4 GGUF).
## First run: creates esp.img (6G FAT32) and copies ALL files (slow, ~1-2 min).
## Subsequent runs: only updates the EFI binary (fast, <5s).
esp-img: esp
	@hdiutil detach /tmp/genos_esp 2>/dev/null || true
	@if [ ! -f $(ESP_IMG) ]; then \
		echo "[esp-img] First build: creating 6G FAT32 disk image (this takes ~1-2 min)..."; \
		dd if=/dev/zero of=$(ESP_IMG) bs=1m count=6144 2>/dev/null; \
		ATTACH_OUT=$$(hdiutil attach -nomount $(ESP_IMG)); \
		DEV=$$(echo "$$ATTACH_OUT" | awk 'NR==1{print $$1}'); \
		echo "[esp-img] Formatting $$DEV as FAT32..."; \
		newfs_msdos -F 32 -v GENOS $$DEV; \
		hdiutil detach $$DEV; \
		hdiutil attach -mountpoint /tmp/genos_esp $(ESP_IMG); \
		echo "[esp-img] Copying files (models may take a while)..."; \
		cp -r $(ESP_DIR)/* /tmp/genos_esp/; \
		hdiutil detach /tmp/genos_esp; \
		echo "[esp-img] $(ESP_IMG) ready (FAT32, 6G)"; \
	else \
		echo "[esp-img] Updating EFI binary in existing esp.img..."; \
		hdiutil attach -mountpoint /tmp/genos_esp $(ESP_IMG); \
		mkdir -p /tmp/genos_esp/EFI/BOOT; \
		cp $(ESP_DIR)/EFI/BOOT/BOOTX64.EFI /tmp/genos_esp/EFI/BOOT/BOOTX64.EFI; \
		hdiutil detach /tmp/genos_esp; \
		echo "[esp-img] EFI binary updated"; \
	fi

## Run in QEMU with OVMF (graphical, release build — fast)
qemu: esp-img
	@if [ "$(OVMF_CODE)" = "OVMF_NOT_FOUND" ]; then \
		echo "ERROR: OVMF firmware not found."; \
		echo "Install QEMU with: brew install qemu (macOS) or apt install ovmf (Linux)"; \
		exit 1; \
	fi
	qemu-system-x86_64 \
		-cpu max \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-drive format=raw,file=$(ESP_IMG) \
		-m 6G \
		-net none \
		-serial stdio

## Run in QEMU with a debug build (slow, for debugging panics)
qemu-debug: esp-img
	$(CARGO) build --target $(TARGET) -p genos-boot $(BUILD_STD)
	@hdiutil detach /tmp/genos_esp 2>/dev/null || true
	@hdiutil attach -mountpoint /tmp/genos_esp $(ESP_IMG)
	@mkdir -p /tmp/genos_esp/EFI/BOOT
	cp target/$(TARGET)/debug/genos-boot.efi /tmp/genos_esp/EFI/BOOT/BOOTX64.EFI
	@hdiutil detach /tmp/genos_esp
	qemu-system-x86_64 \
		-cpu max \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-drive format=raw,file=$(ESP_IMG) \
		-m 6G \
		-net none \
		-serial stdio

## Run in QEMU with user-mode networking (for Phase B HTTP HAL testing)
## Exposes host port 8080 inside the VM as port 80; adds e1000 NIC.
qemu-net: esp-img
	@if [ "$(OVMF_CODE)" = "OVMF_NOT_FOUND" ]; then \
		echo "ERROR: OVMF firmware not found."; \
		exit 1; \
	fi
	qemu-system-x86_64 \
		-cpu max \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-drive format=raw,file=$(ESP_IMG) \
		-m 6G \
		-netdev user,id=net0,hostfwd=tcp::8080-:80 \
		-device e1000,netdev=net0 \
		-serial stdio

## Run in QEMU without graphics (serial console only)
qemu-nographic: esp-img
	@if [ "$(OVMF_CODE)" = "OVMF_NOT_FOUND" ]; then \
		echo "ERROR: OVMF firmware not found."; \
		exit 1; \
	fi
	qemu-system-x86_64 \
		-cpu max \
		-drive if=pflash,format=raw,readonly=on,file=$(OVMF_CODE) \
		-drive format=raw,file=$(ESP_IMG) \
		-m 6G \
		-net none \
		-nographic

## Download Stories15M model and tokenizer
setup-model:
	@mkdir -p $(ESP_DIR)/models
	@echo "Downloading stories15M model..."
	curl -L -o $(ESP_DIR)/models/stories15m.bin \
		"https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
	@echo "Downloading tokenizer (from llama2.c repo)..."
	curl -L -o $(ESP_DIR)/models/tokenizer.bin \
		"https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin"
	@echo "Model files downloaded to $(ESP_DIR)/models/"

## Run integration tests (hosted std test suite in tests/)
test:
	cd tests && cargo test -- --test-threads=1

## Run kernel unit tests on the host machine
## Note: --target is required because .cargo/config.toml defaults to x86_64-unknown-uefi
test-kernel:
	cargo test -p genos-kernel --features hosted --target $(shell rustc -vV | grep host | cut -d' ' -f2)

## Clean build artifacts
clean:
	$(CARGO) clean
	rm -rf $(ESP_DIR) $(ESP_IMG)

## Show help
help:
	@echo "genos build targets:"
	@echo "  make build          - Build the UEFI binary (release)"
	@echo "  make build-debug    - Build the UEFI binary (debug)"
	@echo "  make esp            - Create ESP directory with binary + palace"
	@echo "  make setup-model    - Download Stories15M model + tokenizer"
	@echo "  make qemu           - Run in QEMU (graphical, no network)"
	@echo "  make qemu-net       - Run in QEMU with networking (Phase B)"
	@echo "  make qemu-nographic - Run in QEMU (serial only)"
	@echo "  make test           - Run host-mode unit tests"
	@echo "  make clean          - Clean all artifacts"
	@echo ""
	@echo "First-time setup:"
	@echo "  1. make build"
	@echo "  2. make setup-model"
	@echo "  3. make qemu"
