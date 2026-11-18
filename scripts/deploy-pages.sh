#!/bin/sh
# Builds the site for GitHub Pages and publishes it: `npm run deploy`.
#
# GitHub Pages serves this project at https://<user>.github.io/<repository>/, so the site is built with
# the repository's name as its base path. The result goes to the `gh-pages` branch of `origin` as a single
# commit that replaces the previous one (the branch holds build output only, never history worth keeping).
set -eu
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
	echo "There are uncommitted changes. Commit first, so that the published site matches a commit." >&2
	exit 1
fi

remote=$(git remote get-url origin)
repo=$(basename "$remote" .git)
commit=$(git rev-parse --short HEAD)
name=$(git config user.name)
email=$(git config user.email)

# The owner's name from https://github.com/<owner>/<repository>.git or git@github.com:<owner>/<repository>.git;
# GitHub serves the site from https://<owner, lower case>.github.io.
owner=$(basename "$(dirname "$(echo "$remote" | tr ':' '/')")" | tr '[:upper:]' '[:lower:]')

BASE_PATH="/$repo" SITE_ORIGIN="https://$owner.github.io" npm run build

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cp -R build/. "$out"
cd "$out"
git init -q -b gh-pages
git add -A
git -c user.name="$name" -c user.email="$email" commit -q -m "Deploy $commit"
git push -q -f "$remote" gh-pages

echo "Published $commit to the gh-pages branch of $remote"
