from memory.memory_manager import write_memory

write_memory(
    file_path="memory/profile.json",
    key="test_fact",
    value="this is a safe value",
    source="verification_test"
)

print("✅ Memory write successful")

from memory.memory_manager import auto_memory

auto_memory(
    category="preferences",
    key="emoji_preference",
    value="dislike",
    source="test"
)

print("done")
