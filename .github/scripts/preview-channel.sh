#!/usr/bin/env bash
# Firebase Hosting preview-channel ID for a branch: a lowercase slug of the
# branch name plus a short hash of the full name (unique after truncation).
# Kept short so <site>--<channel>-<hash>.web.app stays a valid DNS label.
#
#   .github/scripts/preview-channel.sh "feature/Hands-v2"   →  feature-hands-v2-1a2b3c
set -euo pipefail

branch="${1:?usage: preview-channel.sh <branch>}"
slug="$(printf '%s' "$branch" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' | cut -c1-32 | sed -E 's/-+$//')"
hash="$(printf '%s' "$branch" | sha1sum | cut -c1-6)"
printf '%s-%s\n' "${slug:-branch}" "$hash"
