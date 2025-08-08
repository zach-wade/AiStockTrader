# Quick Claude Commands - Copy & Paste Ready

## 📄 Update Documentation Command

### Simple Version:
```
Please update @CLAUDE.md and the supporting @CLAUDE-TECHNICAL.md, @CLAUDE-OPERATIONS.md, and @CLAUDE-SETUP.md docs based on any recent changes in the codebase.
```

### Comprehensive Version:
```
Please perform a comprehensive documentation update:

1. First, analyze recent changes:
   - Check the last 10 git commits for code changes
   - Identify modified files and their types
   - Note any new features, commands, or configurations

2. Update the appropriate CLAUDE documentation:
   - CLAUDE.md: Update architecture, commands, and guidelines if core functionality changed
   - CLAUDE-TECHNICAL.md: Update if dependencies, Docker, or technical specs changed
   - CLAUDE-OPERATIONS.md: Update if operational procedures or troubleshooting changed
   - CLAUDE-SETUP.md: Update if setup steps, environment variables, or prerequisites changed

3. For each update:
   - Maintain existing formatting and style
   - Update examples to match current code
   - Fix any broken file paths or references
   - Update version numbers and dates

4. Validate all changes:
   - Ensure cross-references between docs work
   - Verify command syntax is correct
   - Check that file paths exist
   - Confirm configuration examples are accurate

Please show me what changes you're making and why.
```

### Focused Update Commands:

#### After Adding New Features:
```
I've added new features to the codebase. Please update the CLAUDE documentation to reflect these changes, focusing on:
- New commands in CLAUDE.md
- Technical implementation details in CLAUDE-TECHNICAL.md
- Any new operational procedures in CLAUDE-OPERATIONS.md
```

#### After Configuration Changes:
```
I've modified the configuration system. Please update:
- Configuration section in CLAUDE.md
- Environment variables in CLAUDE-SETUP.md
- Any config file references in CLAUDE-TECHNICAL.md
```

#### After Database Changes:
```
I've made database schema changes. Please update:
- Database schema section in CLAUDE.md
- Schema locations in CLAUDE-TECHNICAL.md
- Migration procedures in CLAUDE-OPERATIONS.md
```

#### After Docker/Service Changes:
```
I've modified Docker or service configurations. Please update:
- Docker architecture in CLAUDE-TECHNICAL.md
- Service management in CLAUDE-OPERATIONS.md
- Docker setup in CLAUDE-SETUP.md
```

### Validation Only Command:
```
Please review all CLAUDE documentation files and check for:
- Outdated information that doesn't match the current code
- Broken cross-references between documents
- Incorrect file paths or commands
- Missing documentation for recent features
- Inconsistencies between the different CLAUDE docs

Don't make changes yet, just report what needs updating.
```

### Quick Section Update:
```
Please update the [SPECIFIC SECTION] in @CLAUDE.md based on the current code. The section should reflect [WHAT CHANGED].
```

## 💡 Tips for Using These Commands

1. **Be Specific**: Add context about what changed for better updates
2. **Review Changes**: Always review proposed documentation changes
3. **Test Examples**: Ask Claude to verify code examples work
4. **Batch Updates**: Combine multiple small changes into one update session
5. **Version Control**: Commit documentation updates with clear messages

## 🔄 Suggested Workflow

1. Make code changes
2. Test changes work correctly
3. Run documentation update command
4. Review proposed updates
5. Test documentation examples
6. Commit code and docs separately

---

*Copy and paste these commands directly into your Claude chat for quick documentation updates.*