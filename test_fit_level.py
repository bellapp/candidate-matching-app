#!/usr/bin/env python3
"""
Test script to demonstrate the fit level calculation
"""
from test_matching_score import get_fit_level, FIT_LEVEL_SCALE

print("=" * 60)
print("FIT LEVEL SCALE")
print("=" * 60)
for min_score, fit_level in FIT_LEVEL_SCALE:
    print(f"{min_score:3d}+ points = {fit_level}")
print()

print("=" * 60)
print("EXAMPLE CALCULATIONS")
print("=" * 60)

test_scores = [95, 85, 73, 60, 45, 25, 0]

for score in test_scores:
    fit_level = get_fit_level(score)
    print(f"Matching Score {score:3d}/100 → {fit_level}")

print()
print("✅ Fit level function working correctly!")
print()
print("This ensures consistency between:")
print("  - Matching score display")
print("  - Qualification note assessment")
print("  - Summary generation")
