#!/usr/bin/env bash
# Print the next release version.
#
# Usage: next-version.sh major|minor|patch
#
# The base is the latest vX.Y.Z git tag. With no tags yet, the version in
# pyproject.toml is the first release, whatever the bump.
set -euo pipefail

bump="${1:?usage: next-version.sh major|minor|patch}"

# Only plain release tags: v1.2.3, not v1.2.3rc1
latest=$(git tag --list 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1 || true)

if [ -z "$latest" ]; then
  grep -m 1 '^version = ' pyproject.toml | cut -d '"' -f 2
  exit 0
fi

IFS=. read -r major minor patch <<< "${latest#v}"
case "$bump" in
  major) echo "$((major + 1)).0.0" ;;
  minor) echo "${major}.$((minor + 1)).0" ;;
  patch) echo "${major}.${minor}.$((patch + 1))" ;;
  *) echo "unknown bump: $bump" >&2; exit 1 ;;
esac
