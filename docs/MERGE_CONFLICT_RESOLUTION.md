# Resolving Merge Conflicts in Pull Requests

Follow these steps to manually resolve merge conflicts in a pull request when automatic resolution is not possible:

1. **Switch to your feature branch**
   ```bash
   git checkout codex/implement-lstm-forecaster-with-attention
   ```

2. **Pull the latest changes from `main`**
   ```bash
   git pull origin main
   ```

3. **Identify and fix conflicts**
   - Open the reported files with conflicts.
   - Look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
   - Edit the sections to keep the desired changes and remove the markers once resolved.

4. **Stage the resolved files**
   ```bash
   git add <file-with-resolved-conflicts>
   ```

5. **Commit the merge**
   ```bash
   git commit
   ```

6. **Push the merge commit to the remote branch**
   ```bash
   git push origin codex/implement-lstm-forecaster-with-attention
   ```

After pushing, the pull request should be free of conflicts and ready to merge.
