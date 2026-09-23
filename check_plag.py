import re

# Read the paper
with open('paper/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract sentences (rough)
sentences = re.split(r'(?<=[.!?])\s+', text)

# Filter: longer sentences, not equations, not citations-heavy
candidates = []
for s in sentences:
    s_clean = s.strip()
    if len(s_clean) > 80 and len(s_clean) < 500:
        cite_count = s_clean.count('\\cite')
        word_count = len(s_clean.split())
        if cite_count / max(word_count, 1) < 0.15:
            candidates.append(s_clean)

print(f"Total sentences: {len(sentences)}")
print(f"Candidate sentences for plagiarism check: {len(candidates)}")
print()

candidates.sort(key=len, reverse=True)
for i, c in enumerate(candidates[:30]):
    print(f"{i+1}. ({len(c)} chars) {c[:250]}...")
    print()