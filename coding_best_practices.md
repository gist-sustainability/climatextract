# Coding standards

## General
- Use docstrings for classes and public functions
- comments: use less than we have currently in the code, avoid comments if possible (good code needs very few comments)
- Typing (use it if easy to use)
- Do Code reviews (or pair programming) if it feels helpful

## Using Pylint and PEP8 formatter in VSCode

Pylint
1. Install extension: [Pylint](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)

2. Linters, if installed, are enabled by default. 
3. Linting will automatically run when a Python file is opened or saved. 
4. Errors and warnings are shown in the Problems panel (⇧⌘M) for open files, and are also highlighted in the code editor. Hovering over an underlined issue displays the details.

PEP8
1. Install extension: [PEP8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)
2. Set PEP8 as default formatter for workspace and make formatting automatic on saving a file:
    1. Click ⌘⇧P (^⇧P) to open the Command Palette
    2. Search for `settings.json` and choose the file referring to the Workspace: `Preferences: Open Workspace Settings (JSON)``
    3. Insert this into the JSON object:
    ```
    "[python]": {
      "editor.defaultFormatter": "ms-python.autopep8",
      "editor.formatOnSave": true
    }
    ```

## Commit conventions
- Write your commit message in the imperative: "Fix bug" and not "Fixed bug" or "Fixes bug."
- Use the body to explain what and why you have done something. In most cases, you can leave out details about how a change has been made.
- A properly formed git commit subject line should always be able to complete the following sentence: If applied, this commit will <your subject line here>.

Commit message should be structured as follows:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
````
`<type>`:
- `fix`: a commit of the type fix patches a bug in your codebase
- `feat`: a commit of the type feat introduces a new feature to the codebase
- BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope, introduces a breaking API change 
- types other than fix: `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.

`[scope]`: Example: `feat(lang): add Polish language` 

## Branching

1. After you've selected a feature to work on, create a branch in your local repo to build it in. Use Descriptive Branch Names! 
  `git checkout -b calaway/short_description_of_feature`
2. Implement the requested feature, make sure the code still runs, and commit all changes in the new branch.
3. Checkout the main branch locally.
  `git checkout main`
4. Pull down the main branch from GitHub to get the most up to date changes from others. If you practice git workflow as described here you should never have a merge conflict at this step. Sync the main branch regularly to avoid conflicts.
  `git pull origin main`
5. Make sure all tests are passing on main and then checkout your new branch.
  `git checkout calaway/short_description_of_feature`
6. From your new branch, merge in your local main branch.
  `git merge main`
7. Resolve any merge conflicts, make sure all tests are passing on the new branch, and then commit all changes from the merge.
  `git add .`
  `git commit -m "Merge in main."`
8. Push the feature branch to the remote repo.
  `git push --set-upstream origin calaway/short_description_of_feature`
9. Submit a **pull request** on GitHub asking to merge the branch into main.
10. A teammate reviews the code for quality and functionality.
11. The teammate merges the pull request and deletes your branch from GitHub.
