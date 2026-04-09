# genos — Bare-Metal LLM Operating System
# Build and run targets for UEFI development

CARGO = cargo
TARGET = x86_64-unknown-uefi
BUILD_DIR = target/$(TARGET)/debug
EFI_BINARY = $(BUILD_DIR)/genos-boot.efi
ESP_DIR = esp

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

.PHONY: build esp qemu qemu-nographic clean help setup-model

## Build the UEFI binary
build:
	$(CARGO) build --target $(TARGET) -p genos-boot

## Build in release mode
release:
	$(CARGO) build --target $(TARGET) -p genos-boot --release

## Create the ESP (EFI System Partition) directory structure
esp: build
	@mkdir -p $(ESP_DIR)/EFI/BOOT
	@mkdir -p $(ESP_DIR)/models
	@mkdir -p $(ESP_DIR)/system
	@mkdir -p $(ESP_DIR)/memory
	@mkdir -p $(ESP_DIR)/logs
	@mkdir -p $(ESP_DIR)/data
	@cp $(EFI_BINARY) $(ESP_DIR)/EFI/BOOT/BOOTX64.EFI
	@if [ -f resources/prompt.txt ]; then cp resources/prompt.txt $(ESP_DIR)/system/prompt.txt; fi
	@echo "ESP created at $(ESP_DIR)/"
	@echo ""
	@echo "Before running, place model files in $(ESP_DIR)/models/:"
	@echo "  - stories15m.bin  (model weights)"
	@echo "  - tokenizer.bin   (tokenizer)"
	@echo ""
	@echo "Download from: https://huggingface.co/karpathy/tinyllamas/tree/main"

## Run in QEMU with OVMF (graphical)
qemu: esp
	@if [ "$(OVMF_CODE)" = "OVMF_NOT_FOUND" ]; then \
		echo "ERROR: OVMF firmware not found."; \
		echo "Install QEMU with: brew install qemu (macOS) or apt install ovmf (Linux)"; \
		exit 1; \
	fi
	qemu-system-x86_64 \
		-bios $(OVMF_CODE) \
		-drive format=raw,file=fat:rw:$(ESP_DIR) \
		-m 512M \
		-net none \
		-serial stdio

## Run in QEMU without graphics (serial console only)
qemu-nographic: esp
	@if [ "$(OVMF_CODE)" = "OVMF_NOT_FOUND" ]; then \
		echo "ERROR: OVMF firmware not found."; \
		exit 1; \
	fi
	qemu-system-x86_64 \
		-bios $(OVMF_CODE) \
		-drive format=raw,file=fat:rw:$(ESP_DIR) \
		-m 512M \
		-net none \
		-nographic

## Download Stories15M model and tokenizer
setup-model:
	@mkdir -p $(ESP_DIR)/models
	@echo "Downloading stories15M model..."
	curl -L -o $(ESP_DIR)/models/stories15m.bin \
		"https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
	@echo "Downloading tokenizer..."
	curl -L -o $(ESP_DIR)/models/tokenizer.bin \
		"https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.bin"
	@echo "Model files downloaded to $(ESP_DIR)/models/"

## Run host-mode tests (genos-kernel only, no UEFI dependency)
test:
	cargo test -p genos-kernel --features hosted

## Clean build artifacts
clean:
	$(CARGO) clean
	rm -rf $(ESP_DIR)

## Show help
help:
	@echo "genos build targets:"
	@echo "  make build          - Build the UEFI binary"
	@echo "  make release        - Build in release mode"
	@echo "  make esp            - Create ESP directory with binary"
	@echo "  make setup-model    - Download Stories15M model + tokenizer"
	@echo "  make qemu           - Run in QEMU (graphical)"
	@echo "  make qemu-nographic - Run in QEMU (serial only)"
	@echo "  make test           - Run host-mode unit tests"
	@echo "  make clean          - Clean all artifacts"
	@echo ""
	@echo "First-time setup:"
	@echo "  1. make build"
	@echo "  2. make setup-model"
	@echo "  3. make qemu"
