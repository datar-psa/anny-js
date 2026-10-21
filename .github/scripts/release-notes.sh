#!/usr/bin/env bash
# Release notes for an anny-js tag from Conventional Commit subjects since the
# previous anny-js tag (scopes allowed: "feat(mediapipe): …"). chore/ci/docs/
# test/build/style commits are left out.
#
#   .github/scripts/release-notes.sh anny-js/v0.1.0 > notes.md
set -euo pipefail

tag="${1:?usage: release-notes.sh <tag>}"
prev="$(git tag --list 'anny-js/v*' --sort=-version:refname | grep -vxF "$tag" | head -n1 || true)"
range="${prev:+$prev..}$tag"

section() {  # $1 = title, $2 = type regex
  local lines
  lines="$(git log --format=%s "$range" | sed -nE "s/^($2)(\([^)]*\))?!?: (.*)$/- \3/p")"
  [ -z "$lines" ] || printf '### %s\n%s\n\n' "$1" "$lines"
}

notes="$(section Features feat; section Fixes fix; section Performance perf; section Refactors refactor)"
printf '%s\n' "${notes:-Maintenance release.}"
if [ -n "$prev" ]; then
  printf '\n**Full changelog**: https://github.com/%s/compare/%s...%s\n' "${GITHUB_REPOSITORY:-datar-psa/anny-js}" "$prev" "$tag"
fi
