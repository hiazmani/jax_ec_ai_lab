#!/bin/bash

# Configuration
TARGET_DIR="../jax_ec_ai_lab"
DEPLOY_BRANCH="main"

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo ">>> Exporting clean state from branch [$CURRENT_BRANCH] to $TARGET_DIR..."

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Use git archive to export only tracked files (respects .gitignore)
git archive "$CURRENT_BRANCH" | tar -x -C "$TARGET_DIR"

# Navigate to deployment repo
cd "$TARGET_DIR" || exit

# Check if it is a git repo
if [ ! -d ".git" ]; then
    echo "Error: $TARGET_DIR is not a git repository."
    exit 1
fi

# Commit and Push
echo ">>> Committing and pushing to deployment repository..."
git add .

# Use provided message or default timestamp
COMMIT_MSG=${1:-"Deployment update: $(date +'%Y-%m-%d %H:%M:%S')"}
git commit -m "$COMMIT_MSG"

git push origin "$DEPLOY_BRANCH"

echo ">>> Done! Deployment pushed to AI Lab repo."
