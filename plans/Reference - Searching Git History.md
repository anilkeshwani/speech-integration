# Reference: Searching Git History for Specific Strings or Patterns

## The key tool: `git log -S` (pickaxe search)

`-S "string"` asks git: "show me commits where the number of occurrences of this string changed." This finds commits that **added or removed** the string — not commits where the string merely appears in context. Git doesn't diff every commit sequentially; it uses an optimized internal search over the diff tree.

The `-p` flag shows the actual patch for those commits, so you can see exactly what was added/removed.

## Useful variants

| Command | What it does |
|---------|-------------|
| `git log -S "string"` | Commits where occurrence count of `string` changed (added/removed) |
| `git log -G "regex"` | Commits where a **line matching the regex** was added/removed (more flexible, slightly slower) |
| `git log --all -S "..." -- '*.py'` | Restrict to Python files, search all branches |
| `git log -S "..." --diff-filter=D` | Only show commits that **deleted** files containing the string |
| `git log -- path/to/dir/` | All commits that touched any file in that path |
| `git show <hash>:<path>` | Retrieve the full contents of a file as it existed at a specific commit |

## Efficient workflow

1. **Start with `git log -S "the_thing"`** — this gives you the commits that introduced or removed the reference. Usually just a handful of commits.
2. **Inspect those commits** with `git show <hash>` or use the `-p` flag inline to see the full diff.
3. If you need the full file as it existed at a specific commit: **`git show <hash>:<path>`** — useful for recovering deleted files.
4. For directory-level history: **`git log -- path/to/dir/`** — shows all commits that touched any file in that path.

You never need to iterate over all commits and diff them individually. The pickaxe (`-S`) is the efficient way — git does the heavy lifting internally.

## Example: finding where `prompt_templates` was last used

```bash
# Which commits touched the directory itself?
git log --all --oneline -- prompt_templates/

# Which commits added or removed the string "prompt_templates" in code files?
git log --all -p -S "prompt_templates" --oneline -- '*.py' '*.yaml'

# Recover a deleted file from the commit before it was deleted
git show <commit-before-deletion>:path/to/deleted_file.py
```
