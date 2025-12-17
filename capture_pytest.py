"""Capture pytest output properly."""

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short"],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\Darrin Vogeli\FINAL_GNOSIS",
)

# Write to file
with open("pytest_output.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\n\nExit Code: {result.returncode}")

# Print first 100 lines
lines = (result.stdout + "\n" + result.stderr).split("\n")
for i, line in enumerate(lines[:100], 1):
    print(f"{i}: {line}")

print(f"\n\nTotal lines: {len(lines)}")
print(f"Exit code: {result.returncode}")
