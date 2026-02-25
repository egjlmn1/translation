from deep_translator import GoogleTranslator

# Initialize for English to French
translator = GoogleTranslator(source='en', target='fr')

# Your batch of text
my_batch = [
    "How are you today?",
    "I need this translated quickly.",
    "Python is a great language for automation.",
    "Batch processing is much faster than individual calls."
]

# translate_batch is the key for speed
translated_list = translator.translate_batch(my_batch)

print(translated_list)