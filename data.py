import random
import re
import os
import json
import typing
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
import transformers
from transformers import PreTrainedTokenizer


class GraphPathDataset(Dataset):
    """
    A PyTorch Dataset for graph path-finding experiments.
    
    This dataset is compatible with the MDLM diffusion training pipeline.
    It returns batches with 'input_ids' and 'attention_mask'.
    Labels are NOT needed - the diffusion model handles this internally.
    """
    
    def __init__(
        self,
        path_strings: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        """
        Args:
            path_strings: List of path strings (e.g., "v0 v3 v5 v7")
            tokenizer: The tokenizer to use for encoding
            max_length: Maximum sequence length. If None, uses the longest sequence.
                       Sequences will be padded to this length.
            add_bos: Whether to add BOS token at the start
            add_eos: Whether to add EOS token at the end
        """
        self.path_strings = path_strings
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Pre-tokenize all sequences to determine max length if not provided
        self._tokenized_data = []
        max_seq_len = 0
        
        for path_str in path_strings:
            tokens = self._tokenize_path(path_str)
            self._tokenized_data.append(tokens)
            max_seq_len = max(max_seq_len, len(tokens))
        
        self.max_length = max_length if max_length is not None else max_seq_len
        
    def _tokenize_path(self, path_str: str) -> List[int]:
        """Tokenize a single path string."""
        tokens = []
        
        if self.add_bos and self.tokenizer.bos_token_id is not None:
            tokens.append(self.tokenizer.bos_token_id)
        
        # Tokenize the path string
        encoded = self.tokenizer.encode(path_str, add_special_tokens=False)
        tokens.extend(encoded)
        
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            tokens.append(self.tokenizer.eos_token_id)
            
        return tokens
    
    def __len__(self) -> int:
        return len(self.path_strings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
            - input_ids: Tensor of token IDs, padded to max_length
            - attention_mask: Tensor of 1s for real tokens, 0s for padding
        """
        tokens = self._tokenized_data[idx]
        seq_len = len(tokens)
        
        # Pad or truncate to max_length
        if seq_len < self.max_length:
            # Pad on the right
            pad_length = self.max_length - seq_len
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = tokens + [pad_token_id] * pad_length
            attention_mask = [1] * seq_len + [0] * pad_length
        elif seq_len > self.max_length:
            # Truncate
            input_ids = tokens[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            input_ids = tokens
            attention_mask = [1] * seq_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class GraphPathTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer for graph path-finding experiments.
    
    Fully compatible with HuggingFace ecosystem and MDLM diffusion training.
    Includes all required special tokens: PAD, BOS, EOS, UNK, MASK.
    """
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self,
        vocab_file=None,
        padding_side="right",  # Right padding is standard for training
        **kwargs
    ):
        self.vocab_map = {}
        self.ids_to_tokens_map = {}

        # Define all required special tokens for MDLM diffusion
        # MASK token is important - if not provided, diffusion.py will add one
        default_special_tokens = {
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'mask_token': '[MASK]',
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
        }
        
        # Apply defaults only if not already provided
        for key, value in default_special_tokens.items():
            if key not in kwargs:
                kwargs[key] = value

        kwargs['padding_side'] = padding_side

        # If a vocab_file is provided, load the vocabulary from it.
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab_map = json.load(f)
            self.ids_to_tokens_map = {v: k for k, v in self.vocab_map.items()}
        else:
            # Initialize vocabulary with special tokens in a fixed order
            special_tokens_list = [
                kwargs['cls_token'],
                kwargs['sep_token'],
                kwargs['bos_token'],
                kwargs['eos_token'],
                kwargs['mask_token'],
                kwargs['pad_token'],
                kwargs['unk_token'],
            ]
            self._add_tokens(special_tokens_list)

        # Call the parent's __init__ after the vocabulary is populated.
        super().__init__(**kwargs)
        self.padding_side = padding_side

    def add_special_tokens(self, special_tokens_dict, **kwargs) -> int:
        """
        Adds a dictionary of special tokens and ensures they are added to the vocabulary.
        """
        tokens_to_add = []
        for key, value in special_tokens_dict.items():
            if isinstance(value, list):
                tokens_to_add.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, str):
                tokens_to_add.append(value)

        self._add_tokens(tokens_to_add)
        return super().add_special_tokens(special_tokens_dict, **kwargs)
    
    def add_tokens(self, new_tokens, special_tokens=False) -> int:
        """Add tokens to the vocabulary."""
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens)
    
    def __len__(self):
        return self.vocab_size
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab_map)

    def get_vocab(self) -> Dict[str, int]:
        """Returns a copy of the vocabulary mapping."""
        return self.vocab_map.copy()

    def _add_tokens(self, new_tokens: List[str], **kwargs) -> int:
        """Adds a list of new tokens to the vocabulary."""
        added_count = 0
        for token in new_tokens:
            if isinstance(token, str) and token not in self.vocab_map:
                new_id = len(self.vocab_map)
                self.vocab_map[token] = new_id
                self.ids_to_tokens_map[new_id] = token
                added_count += 1
        return added_count

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenizes a string, correctly handling special tokens.
        """
        # Create a regex pattern that finds any of the special tokens.
        special_tokens_pattern = '|'.join(re.escape(str(token)) for token in self.all_special_tokens_extended)
        pattern = f"({special_tokens_pattern})"

        # Split the text by the special tokens, keeping them in the output.
        chunks = re.split(pattern, text)

        # Process the chunks: keep special tokens, and split regular text by spaces.
        final_tokens = []
        for chunk in chunks:
            if chunk in self.all_special_tokens_extended:
                final_tokens.append(chunk)
            else:
                final_tokens.extend(chunk.strip().split())

        # Filter out any empty strings that may result from the splits.
        return [token for token in final_tokens if token]

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) into an ID using the vocabulary."""
        return self.vocab_map.get(token, self.vocab_map.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) into a token (str) using the vocabulary."""
        return self.ids_to_tokens_map.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        return ' '.join(tokens)
    
    @property
    def token_to_id(self) -> Dict[str, int]:
        """Alias for vocab_map for backward compatibility."""
        return self.vocab_map

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the custom vocabulary to a file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # Use the filename defined in the `vocab_files_names` class attribute.
        vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False, indent=2))

        return (vocab_file,)
    

def build_path_vocab(N, H):
    """
    Builds the vocabulary for the path-finding dataset.

    Args:
        N (int): The number of nodes in the graph.
        H (int): The number of unique seed symbols.

    Returns:
        tuple: A tuple containing lists of nodes, seed tokens, special tokens,
               and the prefixes dictionary.
    """
    NODES = [f"v{i}" for i in range(N)]
    BOS = "<BOS>"
    EOS = "<EOS>"
    prefixes = {
        "train": "path",
        "pretrain": "edge"
    }
    return NODES, BOS, EOS, prefixes

def generate_fixed_degree_dag(nodes, out_degree=3):
    """
    Generates a Directed Acyclic Graph (DAG) where each node attempts to have
    a fixed number of outgoing edges ('out_degree'). This model is much less
    sensitive to its parameters than a probabilistic one.

    Args:
        nodes (list): A list of node names (e.g., ['v0', 'v1', ...]).
        out_degree (int): The desired number of outgoing edges for each node.

    Returns:
        defaultdict: An adjacency list representation of the graph.
    """
    graph = defaultdict(list)
    num_nodes = len(nodes)
    for i in range(num_nodes):
        # The pool of nodes that ui can connect to (uj where j > i) (where [u1, u2, ..., uN] is the list of nodes after shuffling)
        potential_targets = nodes[i+1:]
        
        # If the number of available targets is less than the desired degree,
        # connect to all of them.
        if len(potential_targets) <= out_degree:
            actual_targets = potential_targets
        else:
            # Randomly sample 'out_degree' nodes to connect to.
            actual_targets = random.sample(potential_targets, out_degree)
        
        graph[nodes[i]] = actual_targets
        
    return graph

def generate_hierarchical_dag(nodes, num_layers, nodes_per_layer, edge_prob=0.1, shuffle_nodes=True):
    """
    Generates a Hierarchical Directed Acyclic Graph (DAG) that is guaranteed
    to be connected.

    Args:
        nodes (list): A list of all node names.
        num_layers (int): The total number of layers.
        nodes_per_layer (int): The number of nodes within each layer.
        edge_prob (float): The probability of an edge existing between a node
                           in one layer and a node in the next.
        shuffle_nodes (bool): Parameter included to match the original signature.
                              The algorithm does not use this parameter.

    Returns:
        defaultdict: A defaultdict-based adjacency list representation of the graph.
    """
    if len(nodes) != num_layers * nodes_per_layer:
        raise ValueError("Total number of nodes must equal num_layers * nodes_per_layer")

    # This is the main loop from the algorithm (lines 18-20)
    while True:
        # Initialize an empty graph for this attempt
        graph = defaultdict(list)
        
        # Create the layers from the provided node list
        layers = [nodes[i:i + nodes_per_layer] for i in range(0, len(nodes), nodes_per_layer)]

        # This section implements the CreateLayeredDAG function (lines 7-15)
        for i in range(num_layers - 1):
            current_layer_nodes = layers[i]
            next_layer_nodes = layers[i+1]

            for node in current_layer_nodes:
                for target_node in next_layer_nodes:
                    if random.random() < edge_prob:
                        graph[node].append(target_node)
        
        # This is the connectivity check from the algorithm (line 20)
        # We convert our adjacency list to a NetworkX object temporarily to test it
        temp_nx_graph = nx.DiGraph(graph)
        # Ensure all nodes are in the graph for the connectivity check, even if they have no edges
        temp_nx_graph.add_nodes_from(nodes) 
        
        if nx.is_weakly_connected(temp_nx_graph):
            # If the graph is connected, exit the loop and return the graph
            return graph

def generate_bernoulli_dag(nodes, edge_prob=0.05):
    """
    Generates a random, connected Directed Acyclic Graph (DAG) and returns it
    as an adjacency list.

    The algorithm ensures the graph is acyclic by only allowing edges from lower-indexed
    nodes to higher-indexed nodes. It repeatedly generates graphs until a weakly
    connected one is found.

    Args:
        nodes (list): A list of node names for the graph.
        p (float): The probability of an edge existing between any two valid nodes.

    Returns:
        dict: An adjacency list representation of the connected DAG, where keys
              are node names and values are lists of their neighbors.
    """
    num_nodes = len(nodes)
    if num_nodes <= 0:
        raise ValueError("The list of nodes cannot be empty.")

    # Create a mapping from integer indices to the provided node names
    mapping = {i: name for i, name in enumerate(nodes)}

    # Loop until the generated DAG is connected
    while True:
        # Create a random upper triangular adjacency matrix
        random_matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[1-edge_prob, edge_prob])
        adj_matrix = np.triu(random_matrix, k=1)

        # Create a temporary graph object
        dag = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Check if the graph is weakly connected
        if nx.is_weakly_connected(dag):
            # Relabel the integer nodes to the desired names
            nx.relabel_nodes(dag, mapping, copy=False)
            # Convert to an adjacency list and return
            return nx.to_dict_of_lists(dag)

def tokenize_graph(graph, tokenizer):
    tokenized_graph = {}
    for parent, children in graph.items():
        tokenized_children = [tokenizer.token_to_id[c] for c in children]
        tokenized_graph[tokenizer.token_to_id[parent]] = tokenized_children
    return tokenized_graph

def find_all_paths(graph, start_node, end_node, current_path=None):
    """
    Finds all paths between a start and end node in a DAG using DFS.

    Args:
        graph (dict): The graph's adjacency list.
        start_node (str): The starting node for the path.
        end_node (str): The target end node for the path.
        current_path (list, optional): The path traversed so far. Used for recursion.

    Returns:
        list: A list of lists, where each inner list represents a complete path
              from the start to the end node.
    """
    if current_path is None:
        current_path = []
    
    # Add current node to the path
    path = current_path + [start_node]

    # If we've reached the end, we have found a valid path
    if start_node == end_node:
        return [path]

    # If the start node is not in the graph, no paths exist
    if start_node not in graph:
        return []

    # Recurse for all neighbors
    all_found_paths = []
    for neighbor in graph[start_node]:
        # Continue the search from the neighbor
        new_paths = find_all_paths(graph, neighbor, end_node, path)
        for p in new_paths:
            all_found_paths.append(p)
            
    return all_found_paths

def find_one_random_path(graph, start_node, end_node):
    """
    Finds a single random path from start_node to end_node using a randomized DFS.
    This version corrects the usage of the 'visited' set.
    """
    def dfs(current_node, current_path):
        # Add the current node to the path for this specific recursive exploration
        path = current_path + [current_node]
        
        if current_node == end_node:
            return path

        if current_node not in graph:
            return None

        neighbors = list(graph.get(current_node, []))
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            # The 'path' list passed here correctly tracks visited nodes for the current branch
            if neighbor not in path:
                new_path = dfs(neighbor, path)
                if new_path:
                    return new_path
        
        return None

    return dfs(start_node, [])

def generate_path_strings(graph, nodes, num_samples, EVAL_TOKENIZER, uniform_sampling=True):
    """
    Generates training strings representing paths through the graph.
    Each string is formatted as "start_node end_node start_node ... end_node".

    Args:
        graph (dict): The graph's adjacency list.
        nodes (list): A list of all node names in the graph.
        num_samples (int): The number of path strings to generate.

    Returns:
        list: A list of formatted path strings.
    """
    path_strings = []
    attempts = 0
    max_attempts = num_samples * 5 # Avoid infinite loops in very sparse graphs
    prompts = []
    prompt_to_number_of_paths = defaultdict(int)
    
    while len(path_strings) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample two distinct nodes
        start_node, end_node = random.sample(nodes, 2)

        if EVAL_TOKENIZER.token_to_id[start_node] > EVAL_TOKENIZER.token_to_id[end_node]:
            start_node, end_node = end_node, start_node
        
        if uniform_sampling:
            all_paths = find_all_paths(graph, start_node, end_node)
            if all_paths:
                prompt = f"{start_node} {end_node}"
                prompt_to_number_of_paths[prompt] = len(all_paths)
                # If paths exist, pick one at random
                chosen_path = random.choice(all_paths)
            else:
                chosen_path = None
        # If uniform sampling is not used, find one random path
        else:
            # Find a single random path. This is much more efficient than finding all paths.
            chosen_path = find_one_random_path(graph, start_node, end_node)

        if chosen_path:
            # If a path was found, format the string as "start end path..."
            # which creates the duplicated format, e.g., "v1 v5 v1 v3 v5"
            path_str = " ".join(chosen_path)
            path_strings.append(path_str)
            prompts.append(f"{start_node} {end_node}")
            
    if len(path_strings) < num_samples:
        print(f"Warning: Could only generate {len(path_strings)}/{num_samples} paths. "
              "Consider increasing edge_prob or graph size.")

    return path_strings, prompts, prompt_to_number_of_paths

def make_dataset(N, H, graph_type="bernoulli", edge_prob=0.05, out_degree=4, num_layers=5, pretrain_adjacency=False, num_train_samples=5000, num_test_samples=1000, max_length=None, tokenizer=None, data_root=None, regenerate=False):
    """
    Main function to orchestrate the creation of the path-finding dataset.

    Args:
        N (int): Number of nodes in the graph.
        H (int): Number of seed symbols.
        seed_len (int): Length of the generated seeds.
        edge_prob (float): Probability of an edge in the DAG.
        num_train_samples (int): Number of samples for the training set.
        num_test_samples (int): Number of samples for the test set.
        tokenizer: A tokenizer object to be configured. Must be provided.

    Returns:
        tuple: Contains the train_dataset, test_dataset, raw test strings, and other
               useful artifacts for training and evaluation.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer instance must be provided.")

    # 1. Build Vocabulary
    NODES, BOS, EOS, prefixes = build_path_vocab(N, H)

    # 2. Configure Tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for dataset creation.")
    
    tokenizer.add_special_tokens({'bos_token': BOS})
    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")

    tokenizer.add_special_tokens({'eos_token': EOS})
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

    tokenizer.add_tokens(NODES)
    tokenizer.add_special_tokens({'additional_special_tokens': list(prefixes.values())})

    random.shuffle(NODES)  # Shuffle nodes to ensure randomness in the graph structure
    
    # Create an eval tokenizer with the same vocabulary structure
    EVAL_TOKENIZER = GraphPathTokenizer()
    EVAL_TOKENIZER.add_tokens(NODES)
    EVAL_TOKENIZER.add_special_tokens({'additional_special_tokens': list(prefixes.values())})
    if not data_root:
        raise ValueError("data_root must be specified.")
    
    if graph_type == "bernoulli":
        if edge_prob is None:
            raise ValueError("Edge probability must be specified for Bernoulli graph generation.")
        graph_label = f"graph_{graph_type}-prob{edge_prob}"

    elif graph_type == "fixed-degree":
        if out_degree is None:
            raise ValueError("Out-degree must be specified for fixed-degree graph generation.")
        if out_degree < 1 or out_degree >= N:
            raise ValueError(f"Invalid out-degree {out_degree}. Must be in range [1, {N-1}].")
        if out_degree > N // 2:
            print(f"Warning: Out-degree {out_degree} is high relative to the number of nodes {N}. This may lead to dense connections.")
        graph_label = f"graph_{graph_type}-out{out_degree}"

    elif graph_type == "hierarchical":
        if edge_prob is None or num_layers is None:
            raise ValueError("Edge probability and number of layers must be specified for hierarchical graph generation.")
        graph_label = f"graph_{graph_type}-prob{edge_prob}-L{num_layers}"

    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Supported types are 'bernoulli', 'fixed-degree', and 'hierarchical'.")
    
    data_path = os.path.join(data_root, "path", f"N{N}-{graph_label}") if data_root else None
    if not os.path.exists(data_path):
        regenerate = True
    
    if not regenerate:
        print(f"Loading existing dataset from {data_path}")
        graph = json.load(open(os.path.join(data_path, "graph.json"), "r"))
        train_strs = json.load(open(os.path.join(data_path, "train.json"), "r"))
        test_strs = json.load(open(os.path.join(data_path, "test.json"), "r"))
        train_prompts = json.load(open(os.path.join(data_path, "train_prompts.json"), "r"))
        test_prompts = json.load(open(os.path.join(data_path, "test_prompts.json"), "r"))
        prompt_to_number_of_paths = json.load(open(os.path.join(data_path, "prompt_to_number_of_paths.json"), "r"))

        if len(train_strs) < num_train_samples or len(test_strs) < num_test_samples:
            print(f"Warning: Existing dataset has fewer samples than requested. Regenerating...")
            regenerate = True
            train_strs, train_prompts = [], []
            test_strs, test_prompts = [], []
            prompt_to_number_of_paths = defaultdict(int)
        else:
            train_strs = train_strs[:num_train_samples]
            train_prompts = train_prompts[:num_train_samples]
            test_strs = test_strs[:num_test_samples]
            test_prompts = test_prompts[:num_test_samples]
    
    if regenerate:
        print(f"Generating new dataset with {num_train_samples} training samples and {num_test_samples} test samples...")
        
        # 3. Generate Graph
        if graph_type == "bernoulli":
            graph = generate_bernoulli_dag(NODES, edge_prob=edge_prob)

        elif graph_type == "fixed-degree":
            graph = generate_fixed_degree_dag(NODES, out_degree=out_degree)

        elif graph_type == "hierarchical":
            nodes_per_layer = N // num_layers
            graph = generate_hierarchical_dag(NODES, num_layers=num_layers, nodes_per_layer=nodes_per_layer, edge_prob=edge_prob)
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}. Supported types are 'bernoulli', 'fixed-degree', and 'hierarchical'.")

        # Save the generated graph
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, "graph.json"), "w") as f:
            json.dump(graph, f)

        # 4. Generate Path Strings
        total_samples = num_train_samples + num_test_samples
        strs, prompts, prompt_to_number_of_paths = generate_path_strings(graph, NODES, total_samples, EVAL_TOKENIZER, uniform_sampling=True)
        
        # Split into train and test sets
        train_strs = strs[:num_train_samples]
        test_strs = strs[num_train_samples:]
        train_prompts = prompts[:num_train_samples]
        test_prompts = prompts[num_train_samples:]

        with open(os.path.join(data_path, "train.json"), "w") as f:
            json.dump(train_strs, f)
        with open(os.path.join(data_path, "test.json"), "w") as f:
            json.dump(test_strs, f)
        with open(os.path.join(data_path, "train_prompts.json"), "w") as f:
            json.dump(train_prompts, f)
        with open(os.path.join(data_path, "test_prompts.json"), "w") as f:
            json.dump(test_prompts, f)
        with open(os.path.join(data_path, "prompt_to_number_of_paths.json"), "w") as f:
            json.dump(prompt_to_number_of_paths, f)
        print(f"Successfully generated and saved {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    # # 4. Generate Path Strings
    strs = train_strs + test_strs
    
    assert len(train_strs) == len(train_prompts) == num_train_samples, "Mismatch in number of training strings and prompts."
    assert len(test_strs) == len(test_prompts) == num_test_samples, "Mismatch in number of test strings and prompts."

    # Create datasets compatible with MDLM diffusion training
    train_dataset = GraphPathDataset(
        path_strings=train_strs,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=True,
        add_eos=True,
    )
    test_dataset = GraphPathDataset(
        path_strings=test_strs,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=True,
        add_eos=True,
    )

 
    tokenized_graph = tokenize_graph(graph, EVAL_TOKENIZER)

    print(f"Successfully generated {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    return train_dataset, test_dataset, train_prompts, test_prompts, tokenized_graph, EVAL_TOKENIZER, prefixes, prompt_to_number_of_paths


def get_graph_path_tokenizer(config) -> GraphPathTokenizer:
    """
    Creates and configures a GraphPathTokenizer for use with MDLM.
    
    This function should be called from dataloader.get_tokenizer() 
    when config.data.tokenizer_name_or_path == 'graph_paths'.
    
    Args:
        config: Hydra config object
        
    Returns:
        Configured GraphPathTokenizer
    """
    tokenizer = GraphPathTokenizer()
    
    # Add node tokens based on config
    N = getattr(config.data, 'num_nodes', 100)
    nodes = [f"v{i}" for i in range(N)]
    tokenizer.add_tokens(nodes)
    
    # Add any additional special tokens from prefixes
    prefixes = getattr(config.data, 'prefixes', ['path', 'edge'])
    if prefixes:
        tokenizer.add_special_tokens({'additional_special_tokens': prefixes})
    
    return tokenizer


def get_graph_path_dataloaders(config, tokenizer):
    """
    Creates train and validation dataloaders for graph path-finding.
    
    This function can be called from dataloader.get_dataloaders()
    when config.data.train == 'graph_paths'.
    
    Args:
        config: Hydra config object
        tokenizer: The tokenizer to use
        
    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Extract config parameters
    N = getattr(config.data, 'num_nodes', 100)
    H = getattr(config.data, 'num_seeds', 0)
    graph_type = getattr(config.data, 'graph_type', 'bernoulli')
    edge_prob = getattr(config.data, 'edge_prob', 0.05)
    out_degree = getattr(config.data, 'out_degree', 4)
    num_layers = getattr(config.data, 'num_layers', 5)
    num_train_samples = getattr(config.data, 'num_train_samples', 5000)
    num_test_samples = getattr(config.data, 'num_test_samples', 1000)
    max_length = getattr(config.model, 'length', None)
    data_root = config.data.cache_dir
    regenerate = getattr(config.data, 'regenerate', False)
    
    # Create the datasets
    train_dataset, test_dataset, _, _, _, _, _, _ = make_dataset(
        N=N,
        H=H,
        graph_type=graph_type,
        edge_prob=edge_prob,
        out_degree=out_degree,
        num_layers=num_layers,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        max_length=max_length,
        tokenizer=tokenizer,
        data_root=data_root,
        regenerate=regenerate,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True,
    )
    train_loader.tokenizer = tokenizer
    
    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.loader.eval_batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=False,
    )
    valid_loader.tokenizer = tokenizer
    
    return train_loader, valid_loader