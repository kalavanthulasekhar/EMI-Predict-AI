#!/usr/bin/env python3

with open("app/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line with hero_html markdown
hero_mark_line = None
for i, line in enumerate(lines):
    if 'st.markdown(hero_html, unsafe_allow_html=True)' in line:
        hero_mark_line = i
        break

if hero_mark_line is None:
    print("Could not find hero markdown line")
else:
    # Indent all lines after hero_markdown
    indented_lines = lines[:hero_mark_line + 1]
    
    # Add newline after hero markdown
    indented_lines.append('\n')
    
    # Indent all lines from hero_mark_line+2 onwards
    for i in range(hero_mark_line + 2, len(lines)):
        line = lines[i]
        if line.strip():  # Only indent non-empty lines
            indented_lines.append('    ' + line)
        else:
            indented_lines.append(line)
    
    with open("app/app.py", "w", encoding="utf-8") as f:
        f.writelines(indented_lines)
    
    print(f"✓ Indented {len(lines) - hero_mark_line - 2} lines")
    print(f"✓ Total lines: {len(indented_lines)}")
