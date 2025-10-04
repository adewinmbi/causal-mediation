from nltk.corpus import wordnet as wn
from itertools import chain
from dataclasses import dataclass
from typing import Literal
import random

@dataclass
class DataGenConfig:
    # In the tree, a parent represents the category a child is in.
    
    category_depth: int             # Depth that categories must be from the root word 'entity'
    n_samples: int                  # Number of training samples to make
    n_db_samples: int               # Roughly dictates the variance of your category samples
    n_children: int                 # Roughly dictates the variance of your category children samples
    child_distance: int             # Distance between parent and child. Further distance means category children will be highly specific words.
    max_category_similarity: float  # How similar two categories can be in a single training sample.

def extract_name(title):
    return title.split('.')[0]

def words_with_depth(depth: int, 
                     samples: int=5, 
                     depth_type: str='min', 
                     n_children: int=0, 
                     child_distance: int=1):
    """Get a set of `num_samples` words with the same depth to be used as word types.

    Args:
        depth (int): Depth.
        num_samples (int, optional): Defaults to 5.
        depth_type (str, optional): 'max' or 'min'.
        n_children (int, optional): Number of hyponyms the synset must have.
        child_distance (int, optional): Distance that the child must be from parent.
    """
    data = dict()
    
    for synset in wn.all_synsets(pos='n'):
        d = synset.max_depth() if depth_type == 'max' else synset.min_depth()
        
        if d == depth:
            hypo = lambda s: s.hyponyms()
            children_upto_k = set(synset.closure(hypo, depth=child_distance))
            children_below_k = set(synset.closure(hypo, depth=child_distance-1))
            
            children = list(children_upto_k - children_below_k)[:n_children]
            
            # If the word has at least min_children children, add them
            if len(children) >= n_children:
                data.update({synset: children})
                
            if len(data.keys()) >= samples: 
                return data
            
    return data
    
def gen_data(config: DataGenConfig, ):
    """
    Randomly generate dataset.
    
    Select one randomly from the list of categories, greedily choose 3-8 types. 
    They must not have similarity of above max_category_similarity.
    For each type, randomly choose 1-5 words. Shuffle these together in a list. 
    """
    word_dict = words_with_depth(
        depth=config.category_depth, 
        samples=config.n_db_samples, 
        depth_type='min', 
        n_children=config.n_children, 
        child_distance=config.child_distance)
    
    def _get_category_children(category) -> list:
        potential_children = word_dict[category].copy()
        children = [
            extract_name(potential_children.pop(random.randint(0, len(potential_children)-1)).name())
            for _ in range(random.randint(1, 5)) # Choose 1-5 children per type
            ]
        return children
    
    data = {} # Store: which category is true (str, key), all the fake words (tup), all the true words (tup)
    
    for _ in range(config.n_samples):
        # have set of visited categories
        # don't stop us from revisiting visited categories, BUT, do stop us from going into an infinite loop if we have visited everything
        # fine BC, I expect that we'll have ~100 to ~500 sample space, and we're only looking for 3-8 with a low similarity, so the odds of 
        # a collision happening that many times is low.
        
        true_category = random.choice(list(word_dict.keys()))
        visited = {true_category}
        categories = set()
        
        num_categories = random.randint(2, 7) # Choose 2-7 fake categories
        while len(categories) < num_categories:
            # Greedily find other categories that have a low enough similarity
            candidate = random.choice(list(word_dict.keys()))
            if candidate not in visited:
                if true_category.path_similarity(candidate) <= config.max_category_similarity:
                    categories.add(candidate)
                visited.add(candidate)
        
        data.update({
            extract_name(true_category.name()): (
                _get_category_children(true_category), 
                list(chain(*[_get_category_children(cat) for cat in categories])))
        })
            
    return data
    
if __name__=='__main__':
    # Print samples
    config = DataGenConfig(
        category_depth=4,
        n_samples=3,
        n_db_samples=20,
        n_children=8,
        child_distance=4,
        max_category_similarity=0.15
    )
    
    data = gen_data(config)
    for k, v in zip(data.keys(), data.values()):
        shuffled_list = v[0] + v[1]
        random.shuffle(shuffled_list)
        line = k + '\n' + str(shuffled_list) + '\n' + str(v[0]) + '\n'
        
        print(line)
    