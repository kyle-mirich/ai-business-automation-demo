"""
Script to generate complete Next.js frontend application structure
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "frontend"

# Create directory structure
dirs = [
    "app",
    "app/(dashboard)",
    "app/(dashboard)/rag",
    "app/(dashboard)/support",
    "app/(dashboard)/inventory",
    "components",
    "components/ui",
    "components/features",
    "lib",
    "public",
]

for dir_path in dirs:
    (BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

print("\n✅ Directory structure created!")
print(f"\n📁 Frontend location: {BASE_DIR}")
print("\n🚀 Next steps:")
print("1. cd frontend")
print("2. npm install")
print("3. npm run dev")
print("\nThen open http://localhost:3000")
