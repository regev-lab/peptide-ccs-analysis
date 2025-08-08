import re


def get_relative_positions_of_pattern(peptide, pattern, loc="start", end_offset=-2):
    assert loc in ("start", "end")

    if loc == "start":
        return [m.start() / (len(peptide) + end_offset) for m in re.finditer(pattern, peptide)]
    if loc == "end":
        return [m.end() / (len(peptide) + end_offset) for m in re.finditer(pattern, peptide)]
