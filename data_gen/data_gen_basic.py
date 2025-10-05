import random
from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class SimpleDataGenConfig:
    n_samples: int = 1000           # Number of training samples to generate
    min_list_size: int = 5          # Minimum words in the list
    max_list_size: int = 12         # Maximum words in the list
    min_correct: int = 2            # Minimum correct words in the list
    max_correct: int = 5            # Maximum correct words in the list

# Curated dictionary of common categories with simple, everyday words
CATEGORY_WORDS = {
    "fruit": ["apple", "banana", "orange", "grape", "strawberry", "cherry", "peach", "pear", 
              "watermelon", "mango", "pineapple", "blueberry", "raspberry", "lemon", "lime", 
              "kiwi", "plum", "apricot", "melon", "coconut"],
    
    "animal": ["dog", "cat", "elephant", "lion", "tiger", "bear", "rabbit", "horse", "cow", 
               "sheep", "pig", "chicken", "duck", "goat", "monkey", "zebra", "giraffe", 
               "wolf", "fox", "deer"],
    
    "vehicle": ["car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane", "boat", 
                "ship", "helicopter", "scooter", "taxi", "van", "subway", "tram", "yacht",
                "jet", "rocket", "ambulance", "tractor"],
    
    "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", 
              "brown", "gray", "silver", "gold", "violet", "indigo", "turquoise", "cyan",
              "magenta", "beige", "tan"],
    
    "country": ["USA", "Canada", "Mexico", "Brazil", "UK", "France", "Germany", "Italy", 
                "Spain", "China", "Japan", "India", "Australia", "Russia", "Egypt",
                "Kenya", "Argentina", "Sweden", "Norway", "Thailand"],
    
    "furniture": ["chair", "table", "sofa", "bed", "desk", "couch", "bookshelf", "dresser",
                  "cabinet", "bench", "stool", "ottoman", "nightstand", "wardrobe",
                  "armchair", "recliner", "lamp", "mirror", "rug", "curtain"],
    
    "food": ["pizza", "burger", "pasta", "rice", "bread", "cheese", "salad", "soup", "steak",
             "chicken", "fish", "sandwich", "taco", "sushi", "noodles", "cake", "cookie",
             "pie", "pancake", "waffle"],
    
    "sport": ["soccer", "basketball", "tennis", "baseball", "football", "hockey", "golf",
              "swimming", "running", "cycling", "boxing", "volleyball", "cricket",
              "rugby", "skiing", "surfing", "wrestling", "gymnastics", "badminton", "bowling"],
    
    "instrument": ["guitar", "piano", "violin", "drums", "flute", "trumpet", "saxophone",
                   "clarinet", "cello", "harp", "trombone", "accordion", "harmonica",
                   "banjo", "ukulele", "tuba", "oboe", "xylophone", "bassoon", "mandolin"],
    
    "clothing": ["shirt", "pants", "dress", "skirt", "jacket", "coat", "shoes", "socks",
                 "hat", "gloves", "scarf", "sweater", "jeans", "shorts", "tie", "belt",
                 "boots", "sandals", "vest", "hoodie"],
    
    "tool": ["hammer", "screwdriver", "wrench", "saw", "drill", "pliers", "chisel", "axe",
             "shovel", "rake", "scissors", "knife", "tape", "ruler", "level", "clamp",
             "file", "crowbar", "pickaxe", "mallet"],
    
    "beverage": ["water", "coffee", "tea", "juice", "milk", "soda", "beer", "wine", "coke",
                 "lemonade", "smoothie", "milkshake", "cocktail", "whiskey", "vodka",
                 "champagne", "latte", "espresso", "cappuccino", "mocha"],
    
    "building": ["house", "apartment", "school", "hospital", "church", "temple", "mosque",
                 "tower", "castle", "palace", "barn", "shed", "warehouse", "factory",
                 "mall", "hotel", "restaurant", "library", "museum", "theater"],
    
    "profession": ["doctor", "teacher", "engineer", "lawyer", "nurse", "chef", "pilot",
                   "programmer", "artist", "writer", "musician", "scientist", "farmer",
                   "mechanic", "electrician", "plumber", "architect", "accountant",
                   "dentist", "pharmacist"],
    
    "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "humid",
                "cold", "hot", "warm", "cool", "freezing", "scorching", "mild", "dry",
                "wet", "icy", "hazy", "breezy"],
    
    "emotion": ["happy", "sad", "angry", "excited", "scared", "surprised", "confused",
                "worried", "nervous", "calm", "joyful", "anxious", "proud", "ashamed",
                "grateful", "jealous", "frustrated", "content", "lonely", "relaxed"]
}

def generate_sample(config: SimpleDataGenConfig) -> Tuple[str, List[str], List[str]]:
    """
    Generate a single training sample.
    
    Returns:
        Tuple of (category, mixed_list, correct_words)
    """
    # Choose a random category
    category = random.choice(list(CATEGORY_WORDS.keys()))
    
    # Determine how many correct words to include
    num_correct = random.randint(config.min_correct, config.max_correct)
    num_correct = min(num_correct, len(CATEGORY_WORDS[category]))
    
    # Sample correct words from the chosen category
    correct_words = random.sample(CATEGORY_WORDS[category], num_correct)
    
    # Determine total list size
    total_size = random.randint(config.min_list_size, config.max_list_size)
    num_incorrect = max(0, total_size - num_correct)
    
    # Sample incorrect words from other categories
    other_categories = [cat for cat in CATEGORY_WORDS.keys() if cat != category]
    other_words = []
    for other_cat in other_categories:
        other_words.extend(CATEGORY_WORDS[other_cat])
    
    incorrect_words = random.sample(other_words, min(num_incorrect, len(other_words)))
    
    # Combine and shuffle
    mixed_list = correct_words + incorrect_words
    random.shuffle(mixed_list)
    
    return category, mixed_list, correct_words

def generate_dataset(config: SimpleDataGenConfig) -> List[dict]:
    """
    Generate a complete dataset.
    
    Returns:
        List of dictionaries with keys: 'category', 'list', 'correct_words', 'count'
    """
    dataset = []
    
    for _ in range(config.n_samples):
        category, mixed_list, correct_words = generate_sample(config)
        
        dataset.append({
            "category": category,
            "list": mixed_list,
            "correct_words": correct_words,
            "count": len(correct_words)
        })
    
    return dataset

def save_dataset_txt(dataset: List[dict], filename: str):
    """Save dataset in a simple text format."""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(dataset):
            f.write(f"Type: {sample['category']}\n")
            f.write(f"List: {sample['list']}\n")
            f.write(f"Answer: ({sample['count']})\n")
            f.write(f"Correct words: {sample['correct_words']}\n")
            f.write("\n")

def save_dataset_json(dataset: List[dict], filename: str):
    """Save dataset in JSON format."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

def print_sample_examples(dataset: List[dict], num_examples: int = 5):
    """Print a few examples from the dataset."""
    print(f"Generated {len(dataset)} samples. Here are {num_examples} examples:\n")
    
    for i, sample in enumerate(dataset[:num_examples]):
        print(f"Example {i+1}:")
        print(f"Type: {sample['category']}")
        print(f"List: {sample['list']}")
        print(f"Answer: ({sample['count']})")
        print(f"Correct words: {sample['correct_words']}")
        print()

if __name__ == '__main__':
    # Configure the data generation
    config = SimpleDataGenConfig(
        n_samples=50_000,       # Generate 50k samples
        min_list_size=7,        # Lists will have 7-12 words
        max_list_size=12,
        min_correct=2,          # 2-5 words will be correct
        max_correct=5
    )
    
    # Generate the dataset
    print(f"Generating {config.n_samples} training samples...")
    dataset = generate_dataset(config)
    print_sample_examples(dataset, num_examples=5)
    
    # Save
    save_dataset_json(dataset, "data.json")
    print(f"Dataset saved to 'data.json' (JSON format)")
