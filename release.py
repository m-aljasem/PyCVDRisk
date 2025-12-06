#!/usr/bin/env python3
"""
Release script for CVD Risk Calculator

This script helps with versioning and creating releases.

Usage:
    python release.py [version]

Examples:
    python release.py 0.1.1        # Create release version 0.1.1
    python release.py patch        # Increment patch version
    python release.py minor        # Increment minor version
    python release.py major        # Increment major version
"""

import re
import sys
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def update_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update version
    content = re.sub(r'version = "[^"]*"', f'version = "{new_version}"', content)

    pyproject_path.write_text(content)
    print(f"✅ Updated version to {new_version} in pyproject.toml")


def increment_version(current_version, bump_type):
    """Increment version based on bump type."""
    major, minor, patch = map(int, current_version.split('.'))

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"


def update_changelog(version):
    """Update CHANGELOG.md with new version."""
    changelog_path = Path("CHANGELOG.md")

    if not changelog_path.exists():
        print("⚠️  CHANGELOG.md not found, skipping update")
        return

    content = changelog_path.read_text()

    # Check if version already exists in changelog
    if f"## [{version}]" in content:
        print(f"⚠️  Version {version} already exists in CHANGELOG.md")
        return

    # Add new version entry at the top (after header)
    lines = content.split('\n')
    header_end = 0

    # Find where the header ends (look for first ##)
    for i, line in enumerate(lines):
        if line.startswith('## '):
            header_end = i
            break

    # Insert new version
    new_entry = [
        f"## [{version}] - {get_current_date()}",
        "",
        "### Added",
        "- ",
        "",
        "### Changed",
        "- ",
        "",
        "### Fixed",
        "- ",
        "",
    ]

    lines[header_end:header_end] = new_entry

    changelog_path.write_text('\n'.join(lines))
    print(f"✅ Added version {version} entry to CHANGELOG.md")


def get_current_date():
    """Get current date in YYYY-MM-DD format."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def create_git_tag(version):
    """Create git tag for the release."""
    import subprocess

    try:
        # Create annotated tag
        subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"Release version {version}"], check=True)
        print(f"✅ Created git tag v{version}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create git tag: {e}")
        return False

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python release.py <version|patch|minor|major>")
        print("Examples:")
        print("  python release.py 0.1.1      # Set specific version")
        print("  python release.py patch      # Increment patch version")
        print("  python release.py minor      # Increment minor version")
        print("  python release.py major      # Increment major version")
        sys.exit(1)

    version_arg = sys.argv[1]

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from the project root.")
        sys.exit(1)

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Determine new version
    if version_arg in ["patch", "minor", "major"]:
        new_version = increment_version(current_version, version_arg)
        print(f"Incrementing {version_arg} version: {current_version} → {new_version}")
    else:
        # Direct version specification
        new_version = version_arg
        print(f"Setting version to: {new_version}")

    # Confirm action
    response = input(f"Proceed with version {new_version}? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Release cancelled")
        sys.exit(0)

    # Update version in pyproject.toml
    update_version(new_version)

    # Update changelog
    update_changelog(new_version)

    # Create git commit
    import subprocess
    try:
        subprocess.run(["git", "add", "pyproject.toml", "CHANGELOG.md"], check=True)
        subprocess.run(["git", "commit", "-m", f"Release version {new_version}"], check=True)
        print("✅ Created git commit")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create git commit: {e}")
        sys.exit(1)

    # Create git tag
    if not create_git_tag(new_version):
        sys.exit(1)

    print("🎉 Release preparation complete!")
    print(f"   Version: {new_version}")
    print("   Next steps:")
    print(f"   1. Push to GitHub: git push origin main --tags")
    print(f"   2. GitHub Actions will automatically build and publish to PyPI")
    print(f"   3. Create a GitHub release if needed")


if __name__ == "__main__":
    main()
