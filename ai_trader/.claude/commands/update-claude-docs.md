# /update-claude-docs - Comprehensive Documentation Update Command

## Command: `/update-claude-docs`

### Description
Systematically analyze recent code changes and update all relevant CLAUDE documentation files to reflect the current state of the codebase.

---

## 📋 DOCUMENTATION UPDATE WORKFLOW

### Phase 1: Change Analysis
First, analyze what has changed in the codebase:

```bash
# Check recent commits
git log --oneline -10

# Check uncommitted changes
git status

# Check modified files by type
git diff --name-only | grep -E "\.(py|yaml|sql|md)$" | head -20

# Check for new directories/modules
find src/main -type d -mtime -7 -name "__pycache__" -prune -o -type d -print

# Check for new configuration files
find . -name "*.yaml" -o -name "*.yml" | xargs ls -lt | head -10
```

### Phase 2: Categorize Changes
Identify the type of changes and map to documentation:

| Change Type | Files to Check | Documentation to Update |
|------------|---------------|------------------------|
| **New Features** | `*.py` in `/app/commands/`, `/models/strategies/` | CLAUDE.md (commands), CLAUDE-TECHNICAL.md (architecture) |
| **API Changes** | `/interfaces/`, `/app/` | CLAUDE.md (interfaces), CLAUDE-TECHNICAL.md (patterns) |
| **Config Changes** | `*.yaml`, `.env*` | CLAUDE-SETUP.md (env vars), CLAUDE.md (config section) |
| **Database Changes** | `*.sql`, migrations | CLAUDE-TECHNICAL.md (schema), CLAUDE.md (tables) |
| **Docker/Services** | `docker-compose.yml`, `Dockerfile` | CLAUDE-TECHNICAL.md (docker), CLAUDE-SETUP.md (setup) |
| **Dependencies** | `requirements.txt`, `setup.py` | CLAUDE-TECHNICAL.md (dependencies), CLAUDE-SETUP.md |
| **Operations** | Scripts in `/scripts/`, monitoring | CLAUDE-OPERATIONS.md |
| **Error Handling** | Exception classes, error patterns | CLAUDE-OPERATIONS.md (troubleshooting) |
| **Performance** | Optimization, caching changes | CLAUDE-TECHNICAL.md (performance) |
| **Testing** | `/tests/`, test commands | CLAUDE-SETUP.md (testing), CLAUDE-TECHNICAL.md |

### Phase 3: Update Documentation

#### For CLAUDE.md (Main Reference):
Check and update these sections:
- [ ] **Project Overview** - New capabilities or removed features
- [ ] **System Architecture** - Module changes, new components
- [ ] **Key Systems & Components** - New subsystems or major refactoring
- [ ] **Common Tasks & Commands** - New CLI commands or workflows
- [ ] **File Organization Guide** - New patterns or conventions
- [ ] **Configuration System** - New config files or structure changes
- [ ] **Database Schema** - New tables or schema modifications
- [ ] **Quick Reference** - New environment variables or metrics
- [ ] **Code Review Checklist** - New anti-patterns discovered
- [ ] **Architecture Guidelines** - Updated best practices
- [ ] **Cross-references** - Links to other CLAUDE docs

#### For CLAUDE-TECHNICAL.md:
Check and update these sections:
- [ ] **Language & Runtime** - Dependency version changes
- [ ] **Directory Structure** - New directories or reorganization
- [ ] **Key Code Locations** - New entry points or core files
- [ ] **Docker Architecture** - Container or service changes
- [ ] **Service Ports & Endpoints** - New services or port changes
- [ ] **Development Tools** - New tools or preferences
- [ ] **Coding Style** - Updated conventions
- [ ] **Performance Characteristics** - New benchmarks or targets
- [ ] **File Formats** - New data formats or naming conventions

#### For CLAUDE-OPERATIONS.md:
Check and update these sections:
- [ ] **Service Management** - New start/stop procedures
- [ ] **Database Operations** - New maintenance queries
- [ ] **Log File Locations** - New log files or structure
- [ ] **Monitoring & Metrics** - New metrics or dashboards
- [ ] **Troubleshooting Guide** - New issues and solutions
- [ ] **Data Pipeline Operations** - New backfill or validation procedures
- [ ] **API Testing** - New endpoints or test procedures
- [ ] **Emergency Procedures** - Updated recovery processes
- [ ] **Performance Tuning** - New optimization techniques

#### For CLAUDE-SETUP.md:
Check and update these sections:
- [ ] **Repository Information** - Branch or access changes
- [ ] **Prerequisites** - New system requirements
- [ ] **Python Environment** - Package changes
- [ ] **Environment Variables** - New required variables
- [ ] **Database Initialization** - Schema or migration changes
- [ ] **API Configuration** - New API requirements
- [ ] **Docker Setup** - Container configuration changes
- [ ] **Development Tools** - IDE or tool configuration
- [ ] **Testing Setup** - New test requirements
- [ ] **First Run Checklist** - Updated validation steps

### Phase 4: Specific Update Templates

#### Adding a New Command:
```markdown
# In CLAUDE.md - Common Tasks & Commands section
python ai_trader.py [command] --[options]  # Brief description

# In CLAUDE-TECHNICAL.md - Key Code Locations
- **[Command]**: `src/main/app/commands/[command]_commands.py`
```

#### Adding a New Configuration:
```markdown
# In CLAUDE-SETUP.md - Environment Variables
[VARIABLE_NAME]  # Description and default value

# In CLAUDE.md - Configuration System
├── [new_config].yaml  # Purpose of configuration
```

#### Adding a New Module:
```markdown
# In CLAUDE.md - Module Organization
├── [module_name]/  # Module purpose and responsibility

# In CLAUDE-TECHNICAL.md - Directory Structure
├── src/main/[module_name]/  # Detailed structure
```

### Phase 5: Verification Checklist

#### Cross-Reference Validation:
- [ ] All links between CLAUDE docs work
- [ ] File paths mentioned exist in the codebase
- [ ] Command examples are syntactically correct
- [ ] Configuration examples match actual files

#### Content Accuracy:
- [ ] Version numbers are current
- [ ] Port numbers match docker-compose.yml
- [ ] Environment variables match .env.example
- [ ] Database table names match actual schema
- [ ] API endpoints match implementation

#### Completeness Check:
- [ ] All new features are documented
- [ ] All removed features are deleted from docs
- [ ] All changed behaviors are updated
- [ ] Examples reflect current implementation
- [ ] Troubleshooting covers recent issues

#### Consistency Check:
- [ ] Naming conventions are consistent
- [ ] Code style examples follow actual patterns
- [ ] Technical details align across all docs
- [ ] Dates are updated (Last Updated: YYYY-MM-DD)

### Phase 6: Update Commit Message
```bash
docs: update CLAUDE documentation

- Updated [specific sections] in CLAUDE.md
- Added [new features] to CLAUDE-TECHNICAL.md
- Fixed [outdated info] in CLAUDE-OPERATIONS.md
- Refreshed [setup steps] in CLAUDE-SETUP.md

Changes reflect:
- [List major code changes that triggered updates]
```

---

## 🤖 INTELLIGENT UPDATE PROMPTS

### Prompt 1: Analyze Recent Changes
"Review the git history and working directory changes from the last week. Identify all modifications that would impact user-facing functionality, system architecture, configuration, or operational procedures."

### Prompt 2: Map Changes to Documentation
"Based on the identified changes, determine which CLAUDE documentation files need updating. For each change, specify the exact section and what information needs to be added, modified, or removed."

### Prompt 3: Generate Update Content
"For each documentation update needed, generate the specific markdown content that should be added or modified, ensuring it follows the existing style and format of the documentation."

### Prompt 4: Validate Updates
"Cross-check all documentation updates against the actual code to ensure accuracy. Verify that all examples compile/run and all file paths exist."

### Prompt 5: Check for Completeness
"Review the entire CLAUDE documentation suite to identify any sections that may be outdated or missing information based on the current codebase state."

---

## 🚀 QUICK COMMAND USAGE

To use this command effectively:

1. **Run the command**: `/update-claude-docs`
2. **Let the assistant analyze changes**: It will check recent commits and modifications
3. **Review proposed updates**: The assistant will show what needs updating
4. **Approve updates**: Confirm the changes to be made
5. **Verify results**: Check that all docs are consistent and accurate

---

## 📊 DOCUMENTATION HEALTH METRICS

Track these metrics to ensure documentation quality:

- **Coverage**: Are all features documented?
- **Accuracy**: Do examples and commands work?
- **Freshness**: Are timestamps recent?
- **Consistency**: Do all docs agree?
- **Completeness**: Are all sections filled?
- **Usability**: Can new users follow the docs?

---

## 🔄 AUTOMATED CHECKS

Include these automated checks in the update process:

```python
# Check for broken internal links
grep -r "\[.*\](" CLAUDE*.md | grep -v http | # Check link validity

# Verify code blocks syntax
grep -A5 "```python" CLAUDE*.md | python -m py_compile -

# Check for TODO items
grep -r "TODO\|FIXME\|XXX" CLAUDE*.md

# Verify file paths exist
grep -oP '`[^`]+\.(py|yaml|sql|md)`' CLAUDE*.md | # Verify each path

# Check for outdated version numbers
grep -r "version\|Version" CLAUDE*.md
```

---

## 📝 NOTES FOR EFFECTIVE UPDATES

1. **Be Specific**: Don't just say "updated documentation" - specify what changed
2. **Maintain Style**: Follow existing formatting and conventions
3. **Test Examples**: Ensure all code examples actually work
4. **Update Dates**: Always update the "Last Updated" timestamp
5. **Cross-Reference**: Update related sections across all CLAUDE docs
6. **Think Like a User**: What would someone new to the project need to know?
7. **Document Why**: Not just what changed, but why it matters

---

*This command helps maintain comprehensive, accurate, and useful documentation that truly reflects the current state of the AI Trading System.*