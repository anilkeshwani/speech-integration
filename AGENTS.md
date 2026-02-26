# Repository Guidelines

This is a research codebase intended address the research questions outlined in the Objectives and Key Results.md document in plans/. 

## Notes

Beyond the research goals in plans/, be aware of the following important considerations:

- we are migrating from pip to uv
- we want to clean up the checkpointing, training and other code to make it clean, efficient and express clear intent
- we can make backwards incompatible changes to the codebase in the current refactoring and cleanup phase
    - this includes things like the structure of checkpoints
