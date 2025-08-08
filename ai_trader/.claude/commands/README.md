# Claude Custom Commands

This directory contains custom slash commands for use with Claude AI assistant when working on the AI Trading System project.

## Available Commands

### `/update-claude-docs`
A comprehensive command for updating all CLAUDE documentation files based on recent code changes.

**Usage:**
```
/update-claude-docs
```

This command will:
1. Analyze recent changes in the codebase
2. Identify which documentation needs updating
3. Generate specific update recommendations
4. Validate the documentation for accuracy
5. Ensure cross-references are correct

## How to Use Custom Commands

### Option 1: Direct Command (Quick)
Simply type in your Claude chat:
```
/update-claude-docs

Please analyze recent changes and update @CLAUDE.md, @CLAUDE-TECHNICAL.md, @CLAUDE-OPERATIONS.md, and @CLAUDE-SETUP.md as needed.
```

### Option 2: Comprehensive Command (Thorough)
For a more detailed update process:
```
/update-claude-docs

Please perform a comprehensive documentation update:
1. Check git log for the last 10 commits
2. Identify all changed files and categorize them
3. Update the relevant CLAUDE documentation files
4. Verify all examples and commands still work
5. Update timestamps and version numbers
6. Ensure cross-references between docs are correct
```

### Option 3: Focused Update
For specific changes:
```
/update-claude-docs --focus=[area]

Examples:
- /update-claude-docs --focus=commands  # After adding new CLI commands
- /update-claude-docs --focus=config    # After changing configuration
- /update-claude-docs --focus=docker    # After modifying Docker setup
- /update-claude-docs --focus=api       # After changing APIs or interfaces
```

## Creating New Commands

To add a new custom command:

1. Create a new `.md` file in this directory
2. Name it descriptively (e.g., `check-security.md`)
3. Include:
   - Command name and description
   - Detailed workflow steps
   - Specific prompts for Claude
   - Validation checks
   - Example usage

## Best Practices

1. **Regular Updates**: Run `/update-claude-docs` after significant changes
2. **Before Commits**: Update docs before committing feature changes
3. **Weekly Reviews**: Do a comprehensive doc review weekly
4. **Version Control**: Commit documentation updates separately for clarity

## Command Shortcuts

For frequently used documentation tasks:

- **Quick Update**: `/update-claude-docs --quick` - Only updates changed sections
- **Full Review**: `/update-claude-docs --full` - Complete documentation review
- **Validate Only**: `/update-claude-docs --validate` - Just check for inconsistencies
- **Generate Examples**: `/update-claude-docs --examples` - Update code examples

---

*These commands help maintain high-quality, accurate documentation for the AI Trading System.*