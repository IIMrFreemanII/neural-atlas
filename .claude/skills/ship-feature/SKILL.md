---
name: ship-feature
description: Commit and push a finished feature in this project. Use it without being asked as the last step of every feature, fix or other requested change to the code, tests or docs, once the work is done and verified. Also use it when the user says commit, push or ship. Do not use it for unfinished work, for a failing tree, or for a turn that changed no files.
---

# Ship a finished feature

The owner of this project wants every finished feature committed and pushed without being asked each
time (decided on 2026-11-18). This skill is that standing permission. It covers the steps below and
nothing else: anything in "Ask first" still needs a yes in the conversation.

## When

Run this once per feature, after the feature is complete: the code, its tests and any README change.
One feature is one commit. A fix that a review of the feature turns up belongs in the same commit if it
is not committed yet, and in its own commit if it is.

Do not run it:

- in the middle of a feature, or to save work in progress;
- when a check fails and you have not fixed it;
- for experiments, temporary scripts or changes the user has not asked to keep.

## Steps

### 1. Verify

Run all three and read the results. Do not commit if any of them fails.

```bash
npm run check
```

```bash
npm run test
```

```bash
npm run test:e2e
```

- `check` must report 0 errors and 0 warnings.
- `test:e2e` builds the site, serves it on port 4273 and drives the Chrome installed on this machine.
  In Claude Code it has to run outside the command sandbox.
- A change that touches only Markdown or files under `.claude/` needs none of the three.

If a check fails, fix the cause and run it again. If you cannot fix it, stop and tell the user what
fails, with the output.

### 2. Look at what will be committed

```bash
git status --short
```

```bash
git diff
```

- Stage files by path. Do not use `git add -A` or `git add .`.
- Commit only this feature's files. If the tree holds changes you did not make, leave them out and say
  so.
- Never commit: temporary specs (`tmp-*.spec.ts`), `test-results/`, `.env` files, keys or tokens,
  downloaded datasets, model checkpoints other than the shipped `tinystories.bin`.

### 3. Commit

Commit on the current branch. On this project that is normally `main`: the history is linear, and there
are no feature branches or pull requests unless the user asks for them. If the current branch is not
`main`, commit there and do not merge it into `main` unless the user asks.

The message:

- Subject: imperative, at most 72 characters, says what changed for the learner or the project
  ("Show solved questions on the lesson list and the home map").
- Body: why the change was needed and what is not obvious from the diff. Plain words, lines wrapped at
  72 characters. No list of files.
- Last line: the `Co-Authored-By` line that the session's instructions give.

In Claude Code, git commands that write have to run outside the command sandbox.

### 4. Push

```bash
git remote
```

- No remote named `origin`: do not push and do not create one. The commit stays local. Tell the user
  that it was committed and not pushed, and why. Creating a repository or adding a remote is the user's
  decision.
- Otherwise push the current branch:

```bash
git push origin HEAD
```

- If the push is rejected because the remote has new commits: `git pull --rebase origin <branch>`, run
  `npm run check` and `npm run test` again, then push. Never force.
- If the push fails for another reason (no access, no network), stop and report the error. The commit
  stays local.

### 5. Report

One or two sentences: the commit's short hash and subject, whether it was pushed and where, and the
checks you ran with their results. If something was left out of the commit or the push was skipped, say
that first.

## Ask first

This skill does not cover any of these. Ask in the conversation and wait for a yes:

- a force push, or rewriting commits that are already pushed (amend, rebase, reset);
- skipping the checks, or `--no-verify`;
- creating a repository, adding or changing a remote, or pushing to a remote or branch other than the
  current branch on `origin`;
- tags, releases, deployments;
- committing changes you did not make in this session.
