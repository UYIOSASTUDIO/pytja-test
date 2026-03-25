#!/bin/bash
set -e

echo "--- Installing Pytja Enterprise CLI ---"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [ "$OS" = "darwin" ]; then
    TARGET="pytja-macos.zip"
    BIN_NAME="pytja-macos"
elif [ "$OS" = "linux" ]; then
    TARGET="pytja-linux.zip"
    BIN_NAME="pytja-linux"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

DOWNLOAD_URL="https://github.com/DEIN_GITHUB_NAME/pytja/releases/latest/download/$TARGET"
TMP_DIR=$(mktemp -d)
ZIP_PATH="$TMP_DIR/$TARGET"

echo "Downloading Pytja from $DOWNLOAD_URL..."
curl -sSL --fail "$DOWNLOAD_URL" -o "$ZIP_PATH"

echo "Extracting archive..."
unzip -q "$ZIP_PATH" -d "$TMP_DIR/pytja-release"

echo "Installing to /usr/local/bin (requires sudo privileges)..."
sudo mv "$TMP_DIR/pytja-release/$BIN_NAME" /usr/local/bin/pytja
sudo chmod +x /usr/local/bin/pytja

rm -rf "$TMP_DIR"

echo "Installation successful! Run 'pytja --help' to get started."