**What this changes**

**How I checked it**

- [ ] `ruff check --select E9,F63,F7,F82 .`
- [ ] `python -m pytest tests -q -p no:cacheprovider` in `package/soccer_player_recognition` (no new failures)
