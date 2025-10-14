#!/usr/bin/env python3
"""Test runner script for PyVolGrid.

This script provides convenient ways to run different types of tests.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print(f"✅ {description} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <command>")
        print("\nAvailable commands:")
        print("  all          - Run all tests (excluding slow ones)")
        print("  fast         - Run fast tests only")
        print("  slow         - Run slow tests only")
        print("  coverage     - Run tests with coverage report")
        print("  validation   - Run input validation tests only")
        print("  calculation  - Run volume calculation tests only")
        print("  edge         - Run edge case tests only")
        print("  flexible     - Run flexible interface tests only")
        print("  arrays       - Run array requirements tests only")
        print("  scalar       - Run scalar radius tests only")
        print("  single <test> - Run a single test")
        return 1

    command = sys.argv[1].lower()

    if command == "all":
        success = run_command(
            ["uv", "run", "pytest", "-v", "-m", "not slow"],
            "All tests (excluding slow)",
        )

    elif command == "fast":
        success = run_command(
            ["uv", "run", "pytest", "-v", "-m", "not slow"], "Fast tests only"
        )

    elif command == "slow":
        success = run_command(
            ["uv", "run", "pytest", "-v", "-m", "slow"], "Slow tests only"
        )

    elif command == "coverage":
        success = run_command(
            [
                "uv",
                "run",
                "pytest",
                "--cov=pyvolgrid",
                "--cov-report=html",
                "--cov-report=term-missing",
                "-v",
                "-m",
                "not slow",
            ],
            "Tests with coverage",
        )
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")

    elif command == "validation":
        success = run_command(
            ["uv", "run", "pytest", "tests/test_input_validation.py", "-v"],
            "Input validation tests",
        )

    elif command == "calculation":
        success = run_command(
            ["uv", "run", "pytest", "tests/test_volume_calculation.py", "-v"],
            "Volume calculation tests",
        )

    elif command == "edge":
        success = run_command(
            ["uv", "run", "pytest", "tests/test_edge_cases.py", "-v", "-m", "not slow"],
            "Edge case tests",
        )

    elif command == "flexible":
        success = run_command(
            ["uv", "run", "pytest", "tests/test_flexible_interface.py", "-v"],
            "Flexible interface tests",
        )

    elif command == "arrays":
        success = run_command(
            ["uv", "run", "pytest", "tests/test_array_requirements.py", "-v"],
            "Array requirements tests",
        )

    elif command == "scalar":
        success = run_command(
            ["uv", "run", "pytest", "-v", "-k", "scalar_radius"],
            "Scalar radius tests",
        )

    elif command == "single":
        if len(sys.argv) < 3:
            print("Usage: python run_tests.py single <test_name>")
            print(
                "Example: python run_tests.py single test_single_sphere_volume_approximation"
            )
            return 1

        test_name = sys.argv[2]
        success = run_command(
            ["uv", "run", "pytest", "-v", "-k", test_name], f"Single test: {test_name}"
        )

    else:
        print(f"Unknown command: {command}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
