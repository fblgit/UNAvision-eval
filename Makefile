# UNAvision snapshot unpack
#
# unavision.zip ships the sensitive artifacts as .enc files:
#   modeling_una_ema.py.enc
#   train_sft_ema.py.enc
#   UNA-16x-encoder-decoder.pth.enc
#   UNA-sft-v46.pth.enc
#
# Usage:
#   make unpack   # extract zip and decrypt every .enc in place
#   make clean    # remove decrypted artifacts (keeps .enc and zip)
#
# Passphrase is prompted once on /dev/tty; never stored on disk.

ZIP         := unavision.zip
ENC_CIPHER  := -aes-256-cbc -pbkdf2 -iter 200000 -salt

.PHONY: unpack extract decrypt clean

unpack: extract decrypt

extract:
	@if [ ! -f "$(ZIP)" ]; then echo "error: $(ZIP) not found"; exit 2; fi
	@echo "Extracting $(ZIP)..."
	@unzip -o -q "$(ZIP)"

decrypt:
	@set -eu; \
	shopt -s nullglob 2>/dev/null || true; \
	files=$$(ls *.enc 2>/dev/null || true); \
	if [ -z "$$files" ]; then echo "no .enc files present"; exit 0; fi; \
	stty -echo 2>/dev/null || true; \
	printf 'Passphrase: ' > /dev/tty; IFS= read -r PASS < /dev/tty; echo > /dev/tty; \
	stty echo 2>/dev/null || true; \
	if [ -z "$$PASS" ]; then echo "error: empty passphrase"; exit 3; fi; \
	n=0; \
	for f in $$files; do \
		out="$${f%.enc}"; \
		if [ -e "$$out" ]; then echo "skip: $$out already exists"; continue; fi; \
		if ! printf '%s\n' "$$PASS" | openssl enc -d $(ENC_CIPHER) -pass stdin -in "$$f" -out "$$out" 2>/dev/null; then \
			rm -f "$$out"; echo "error: decrypt failed on $$f (bad passphrase?)"; exit 4; \
		fi; \
		echo "  decrypted $$f -> $$out"; \
		n=$$((n+1)); \
	done; \
	unset PASS; \
	echo "decrypted $$n file(s)"

clean:
	@for f in *.enc; do \
		[ -f "$$f" ] || continue; \
		out="$${f%.enc}"; \
		[ -f "$$out" ] && rm -f "$$out" && echo "removed $$out" || true; \
	done
