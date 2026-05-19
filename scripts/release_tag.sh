#!/usr/bin/env bash
set -euo pipefail

# Create and push an annotated git tag for a release.
# Usage:
#   scripts/release_tag.sh                # uses version from pyproject.toml
#   scripts/release_tag.sh 0.0.1          # explicit version

PROJECT_NAME="napari-harpy"
PYPROJECT_FILE="pyproject.toml"
REMOTE="origin"
RELEASE_BRANCH="${RELEASE_BRANCH:-main}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must be run inside a git repository." >&2
  exit 1
fi

if [[ ! -f "${PYPROJECT_FILE}" ]]; then
  echo "Error: ${PYPROJECT_FILE} not found." >&2
  exit 1
fi

PYPROJECT_NAME="$(awk -F'"' '/^name = "/ { print $2; exit }' "${PYPROJECT_FILE}")"
if [[ "${PYPROJECT_NAME}" != "${PROJECT_NAME}" ]]; then
  echo "Error: expected project '${PROJECT_NAME}', found '${PYPROJECT_NAME}' in ${PYPROJECT_FILE}." >&2
  exit 1
fi

if ! git remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "Error: git remote '${REMOTE}' does not exist." >&2
  exit 1
fi

VERSION="${1:-}"
if [[ -z "${VERSION}" ]]; then
  VERSION="$(awk -F'"' '/^version = "/ { print $2; exit }' "${PYPROJECT_FILE}")"
fi

if [[ -z "${VERSION}" ]]; then
  echo "Error: could not determine version (pass it as first argument)." >&2
  exit 1
fi

TAG="v${VERSION}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ "${CURRENT_BRANCH}" != "${RELEASE_BRANCH}" ]]; then
  echo "Warning: current branch is '${CURRENT_BRANCH}', not '${RELEASE_BRANCH}'." >&2
  echo "Aborting to avoid tagging the wrong commit." >&2
  exit 1
fi

git fetch "${REMOTE}" "${RELEASE_BRANCH}" --tags >/dev/null

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "${REMOTE}/${RELEASE_BRANCH}")"
if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  echo "Error: local HEAD does not match ${REMOTE}/${RELEASE_BRANCH}." >&2
  echo "Push or pull before creating a release tag." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash changes first." >&2
  exit 1
fi

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "Error: local tag '${TAG}' already exists." >&2
  exit 1
fi

if git ls-remote --tags "${REMOTE}" "refs/tags/${TAG}" | grep -q .; then
  echo "Error: remote tag '${TAG}' already exists on '${REMOTE}'." >&2
  exit 1
fi

echo "Creating annotated tag '${TAG}' on commit $(git rev-parse --short HEAD)"
git tag -a "${TAG}" -m "Release ${VERSION}"

echo "Pushing '${TAG}' to '${REMOTE}'"
git push "${REMOTE}" "${TAG}"

cat <<EOF
Done.

Next step: create and publish a GitHub Release for '${TAG}'.
Publishing that GitHub Release triggers .github/workflows/release.yaml and uploads '${PROJECT_NAME}' to PyPI.
EOF
