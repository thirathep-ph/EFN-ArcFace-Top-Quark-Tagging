"""Fix the generate_notebook.py file - replace string ']' and '})' with actual code."""
with open('generate_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The problematic section is:
# "    return fig\n",
# "    ]\n",
# "})\n",
# 
# It should be:
# "    return fig\n",
#     ]
# })

# Find and replace the exact pattern
old = '"    return fig\\n",\n"    ]\\n",\n"})\\n",\n\n"# Cell 14:'
new = '"    return fig\\n",\n    ]\n})\n\n"# Cell 14:'

if old in content:
    content = content.replace(old, new)
    with open('generate_notebook.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed successfully!')
else:
    print('Pattern not found. Checking what is there...')
    # Find the area around 'return fig'
    idx = content.find('return fig')
    if idx >= 0:
        print(content[idx:idx+200])
