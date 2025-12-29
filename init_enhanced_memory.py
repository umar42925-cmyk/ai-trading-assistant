# init_enhanced_memory.py
"""
Initialize enhanced memory system.
Run this once to set up the new features.
"""
import os
import sqlite3
from memory.vector_memory import VectorMemory
from memory.personality_engine import PersonalityEngine
from memory.pattern_recognizer import PatternRecognizer

def initialize_all():
    print("Initializing Enhanced Memory System...")
    
    # Create memory directory if not exists
    os.makedirs("memory", exist_ok=True)
    
    # Initialize vector memory (will create SQLite DB)
    print("1. Setting up vector database and SQLite...")
    vm = VectorMemory()
    print("   ✓ Vector memory initialized")
    
    # Initialize personality engine
    print("2. Setting up personality engine...")
    pe = PersonalityEngine()
    print("   ✓ Personality engine initialized")
    
    # Initialize pattern recognizer
    print("3. Setting up pattern recognition...")
    pr = PatternRecognizer()
    print("   ✓ Pattern recognizer initialized")
    
    # Create default files if they don't exist
    default_files = {
        "memory/personality_profile.json": {
            "traits": {},
            "communication_style": {
                "formality": 0.5,
                "detail_level": 0.6,
                "emotional_tone": 0.5,
                "humor_level": 0.3
            },
            "decision_patterns": [],
            "preferred_topics": [],
            "last_updated": "2025-01-01T00:00:00"
        },
        "memory/behavior_patterns.json": {
            "daily_patterns": {},
            "weekly_patterns": {},
            "topic_patterns": {},
            "mood_patterns": {},
            "conversation_patterns": {}
        }
    }
    
    for file_path, default_content in default_files.items():
        if not os.path.exists(file_path):
            import json
            with open(file_path, "w") as f:
                json.dump(default_content, f, indent=2)
            print(f"   ✓ Created {file_path}")
    
    print("\n✅ Enhanced memory system initialized successfully!")
    print("\nNext steps:")
    print("1. Run your main application")
    print("2. Use the new sidebar features in app.py")
    print("3. The system will learn and improve over time")

if __name__ == "__main__":
    initialize_all()