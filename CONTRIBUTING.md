# Contributing

I try to reply to issues within a few days.

## Setup

```bash
git clone https://github.com/jayhemnani9910/soccer-vision-research
cd soccer-vision-research
python -m venv .venv && source .venv/bin/activate
pip install -e package/soccer_player_recognition pytest psutil ruff==0.17.4
pip install -r yt/requirements.txt   # only if you work on yt/
```

A GPU is not needed for the tests. CPU PyTorch is enough.

## Tests and lint

```bash
ruff check --select E9,F63,F7,F82 .
cd package/soccer_player_recognition && python -m pytest tests -q -p no:cacheprovider
```

The suite has 62 known failures, mostly tests for APIs that were never written. A PR should not add new failures. Fixing one of the old ones, either the code or a test that tests nothing real, is welcome.

## Style

Match the code around your change. Keep PRs small and about one thing. If you add a demo, say in its output when the numbers come from synthetic data.

## Branches and PRs

Branch from `master`, open a PR against `master`, and say what you ran to check it. CI runs lint, a yt smoke check and the tests on every push.

## Bug reports

Open an issue with the steps, what you expected, what happened, your OS, Python and PyTorch versions, and whether you used a GPU.
