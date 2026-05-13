# Git & GitHub workflow — starter guide

This is your reference for our daily git workflow on the summer project. Bookmark it. Refer back to it. You won't memorize all of this in one sitting and that's fine.

If you hit something not covered here, **post in `#tech-help`**. Don't sit on it. Git problems are way easier to fix at minute 5 than at minute 50.

---

## What git is, in 30 seconds

Git is a tool that tracks every change to every file in a project, lets multiple people work on the same code without stepping on each other, and makes it possible to undo almost anything. GitHub is a website that hosts git repositories and adds collaboration features (pull requests, code review, issues).

The mental model: your code lives in three places.
1. **Your working directory** — the files you see and edit
2. **Your local repo** — the git history on your machine
3. **The remote (GitHub)** — the shared copy everyone pulls from

You move changes between these with specific commands. Most confusion early on comes from not knowing which place is which.

---

## One-time setup

### Install git

- macOS: probably already installed. Check with `git --version`. If not: `brew install git` or download from [git-scm.com](https://git-scm.com).
- Linux: `sudo apt install git` (or your distro's equivalent).
- Windows: install [Git for Windows](https://git-scm.com/download/win). Use Git Bash as your terminal.

### Tell git who you are

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use the same email that's on your GitHub account.

### Set up authentication with GitHub

Two options — pick one.

**HTTPS + Personal Access Token (faster setup):**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token, give it `repo` scope, set an expiration (90 days is fine)
3. Copy the token, save it somewhere safe (you won't see it again)
4. First time you push, git asks for username and password — use your GitHub username and paste the **token** as the password
5. Git's credential helper will remember it

**SSH (nicer long-term):**
GitHub has a good guide: [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh). The summary is: generate a key with `ssh-keygen -t ed25519`, add the public key to your GitHub account, test with `ssh -T git@github.com`.

### Clone the repo

```bash
# HTTPS
git clone https://github.com/<org>/fluq.git

# SSH
git clone git@github.com:<org>/fluq.git

cd fluq
```

You now have a local copy of the repo.

---

## The everyday workflow

This is the loop you'll do over and over. Learn this and you're 90% of the way there.

### 1. Start from an up-to-date main

Every time you sit down to work, **first** sync your local main with the remote:

```bash
git switch main
git pull
```

If `git switch` doesn't work, you have an old git version — use `git checkout main` instead.

### 2. Make a branch

Never work directly on main. Make a branch for whatever you're doing:

```bash
git switch -c team2/ensemble-class-skeleton
```

(Or `git checkout -b team2/ensemble-class-skeleton` on older git.)

Branch naming convention from the kickoff doc:
- `team2/...` for Deep Ensembles work
- `team3/...` for Bias/Variance work
- `common/...` for shared utilities
- Be descriptive: `team2/add-heteroscedastic-head` is better than `team2/changes`

### 3. Edit files, then commit

Do your work. When you've made a meaningful chunk of progress (working state, even if not complete), commit it:

```bash
git status                          # see what changed
git add path/to/file.py             # stage specific files
# or
git add .                           # stage everything (be careful!)
git commit -m "Add fit() skeleton to DeepEnsemble class"
```

**Commit early, commit often.** Small commits are easy to review and easy to undo. Don't save up a week of work into one giant commit.

**Commit message style:**
- Imperative mood: "Add ensemble class" not "Added ensemble class" or "Adding ensemble class"
- First line under ~60 characters
- If you need more detail, leave a blank line and write a paragraph
- Don't write `"fix"` or `"changes"` or `"asdf"`. Future-you will hate present-you.

### 4. Push your branch

```bash
git push -u origin team2/ensemble-class-skeleton
```

The `-u` is only needed the first time you push a new branch — it tells git "track this branch on the remote." After that, just `git push`.

### 5. Open a pull request

Go to GitHub in your browser. It'll usually show a banner saying "Your branch had recent pushes — Compare & pull request." Click it. If not, go to the Pull Requests tab and open one manually.

In the PR:
- **Title:** what this PR does, in one line
- **Description:** why you're doing it, anything reviewers need to know to test it, any open questions
- Request a review from whoever should look at it (your team lead, or me)

Keep PRs **small and focused**. One feature, one bug fix, one refactor. A 2000-line PR is unreviewable; a 200-line PR gets real feedback.

### 6. Address review comments

Reviewer leaves comments. You make changes locally, commit, and push to the same branch — the PR updates automatically.

```bash
# make changes
git add .
git commit -m "Address review: rename predict_uq to predict_with_uncertainty"
git push
```

### 7. Merge

Once approved, merge via the GitHub UI ("Merge pull request" button). Use **"Squash and merge"** by default — it collapses your branch's commits into one clean commit on main. Less noise in the history.

After merging, delete the branch (GitHub offers a button) and clean up locally:

```bash
git switch main
git pull
git branch -d team2/ensemble-class-skeleton
```

---

## Things that will go wrong, and what to do

### "I committed to main by accident"

```bash
# Move your most recent commit to a new branch
git switch -c team2/whatever-you-meant
# Reset main back to where it was before
git switch main
git reset --hard origin/main
```

If you've already pushed to main: stop, post in `#tech-help`. We can fix it but I want to make sure we don't lose work.

### "My push was rejected because the remote has changes"

Someone else pushed to your branch (or to main) before you. Pull their changes, then push yours:

```bash
git pull
git push
```

If git complains about a merge or rebase, see the next entry.

### "I have merge conflicts"

This happens when two people changed the same lines. Git can't decide which version to keep, so it asks you.

When you pull or merge and git says "CONFLICT in path/to/file.py":

1. Open the conflicting file. You'll see ugly markers like:
   ```
   <<<<<<< HEAD
   my version of the code
   =======
   their version of the code
   >>>>>>> branch-name
   ```
2. Edit the file to keep what you actually want. **Delete the `<<<<<<<`, `=======`, and `>>>>>>>` lines.**
3. Save the file.
4. `git add path/to/file.py`
5. `git commit` (no message needed, git fills one in)
6. `git push`

If the conflict is in something tricky and you're not sure what to keep, **stop and ask in `#tech-help`**. Don't guess.

### "I want to undo my last commit but I haven't pushed yet"

```bash
# Keep the changes, just undo the commit
git reset --soft HEAD~1

# Throw away the changes entirely
git reset --hard HEAD~1   # CAREFUL: this is destructive
```

### "I want to update my branch with the latest main"

Your branch is getting stale because main moved on. Catch up:

```bash
git switch main
git pull
git switch your-branch
git merge main
```

If that creates conflicts, resolve them like above.

### "I accidentally committed a big file / dataset / secret"

**Stop. Don't push.** Post in `#tech-help`. Removing things from git history is annoying but possible — the longer you wait and the further the bad commit travels, the harder it gets.

To prevent this in the first place: see `.gitignore` below.

### "Git asks me for my password every push"

Your credential helper isn't caching. On macOS:
```bash
git config --global credential.helper osxkeychain
```
On Windows, the Git Bash installer usually sets this up. On Linux:
```bash
git config --global credential.helper cache
```
Or just switch to SSH.

---

## `.gitignore`

There are files you should **never** commit:
- `__pycache__/`, `*.pyc` (Python bytecode)
- `.ipynb_checkpoints/` (Jupyter auto-saves)
- `.env`, `.venv/`, `venv/` (virtual environments)
- `.DS_Store` (macOS junk)
- `*.h5`, `*.ckpt`, `*.pt` (saved model weights — usually too big)
- Any dataset files
- Anything with credentials

The repo will have a `.gitignore` at the root that handles most of this. If you find yourself about to commit a file that probably shouldn't be in git, ask first.

**Datasets stay out of the repo.** We load them via pyMAISE or from a shared location. Don't commit a 500MB CSV.

---

## Notebooks: a special note

Jupyter notebooks are mostly fine in git, but they have one annoying property: the file changes every time you run a cell, even if you didn't change any code. This makes diffs and merges painful.

Conventions for this repo:
- **Clear all outputs before committing** a notebook (`Cell → All Output → Clear` in classic Jupyter, or `Edit → Clear All Outputs` in JupyterLab).
- If a notebook is meant to be a "results" artifact with plots embedded, that's fine — but say so in the commit message.
- Don't put core code in notebooks. Notebooks call into the Python package in `deep_ensembles/` or `bias_variance/`. The package is the real codebase; notebooks are demonstrations.

---

## When to ask for help

Post in `#tech-help` if:
- A command did something you didn't expect
- You have a merge conflict you're not sure how to resolve
- You think you may have lost work
- The instructions here don't match what you're seeing
- You just want a sanity check before running something destructive

Don't:
- Run `git reset --hard`, `git push --force`, or anything with `--force` on a shared branch without asking
- Try to "fix" history with rebase if you've never used rebase before
- Delete `.git/` and re-clone to escape a problem (you'll lose your uncommitted work)

When in doubt: **commit your current state to a branch, push it, then ask.** Once it's pushed, it's basically impossible to lose.

---

## Recommended reading (optional)

If you want to actually understand git rather than just memorize commands:
- [Learn Git Branching](https://learngitbranching.js.org/) — interactive visual tutorial, takes about an hour
- [Pro Git book](https://git-scm.com/book/en/v2), chapters 1-3 — free online, comprehensive

You don't need either of these to do the work this summer. But they will pay back the time investment over the rest of your career.