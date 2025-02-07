import pickle
from collections import deque
from pyvis.network import Network
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm  # Import tqdm for progress tracking
from src.relation_extraction import from_text_to_kb  # Import the function
from src.knowledge_base import KB  # Import the KB class

def save_knowledge_graph(kb, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(kb, file)

def load_knowledge_graph(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def process_multiple_texts_to_kb(list_of_texts, span_length=1024, verbose=False):
    main_kb = KB()
    # Ensure documents are unique
    unique_texts = {text["text"]: text for text in list_of_texts}.values()
    for i, text in enumerate(tqdm(unique_texts, desc="Processing texts", unit="text")):
        if verbose:
            print(f"\nProcessing text {i + 1} of {len(unique_texts)}...")
        kb = from_text_to_kb(text["text"], span_length=span_length, verbose=verbose)  # Ensure text is a string
        for relation in kb.relations:
            if not main_kb.exists_relation(relation):
                main_kb.add_relation(relation)
    return main_kb

def visualize_knowledge_graph(kb_data):
    kg_net = Network(notebook=True, directed=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
    added_nodes = set()
    for relation in kb_data:
        head = relation['head']
        tail = relation['tail']
        relation_type = relation['type']
        if head not in added_nodes:
            kg_net.add_node(head, label=head, title=head)
            added_nodes.add(head)
        if tail not in added_nodes:
            kg_net.add_node(tail, label=tail, title=tail)
            added_nodes.add(tail)
        kg_net.add_edge(head, tail, label=relation_type, title=relation_type)
    kg_net.repulsion(node_distance=200, central_gravity=0.2, spring_length=150, spring_strength=0.05, damping=0.1)
    kg_net.hrepulsion(node_distance=200)
    kg_net.show_buttons(filter_=['physics'])
    kg_net.save_graph("knowledge_graph.html")

def bfs_multi_hop_reasoning(kb, start_entity, max_hops=3):
    visited = set()
    queue = deque([(start_entity, 0)])
    related_info = {}
    while queue:
        current_entity, hop_count = queue.popleft()
        if hop_count > max_hops:
            break
        if current_entity in visited:
            continue
        visited.add(current_entity)
        for relation in kb.relations:
            if relation["head"] == current_entity:
                tail = relation["tail"]
                relation_type = relation["type"]
                if current_entity not in related_info:
                    related_info[current_entity] = []
                related_info[current_entity].append({
                    "relation": relation_type,
                    "tail": tail
                })
                if tail not in visited:
                    queue.append((tail, hop_count + 1))
    return related_info

def collect_kg_info(entities, combined_kb):
    kg_info = {}
    for entity in entities:
        related_info = bfs_multi_hop_reasoning(combined_kb, entity, max_hops=3)
        for entity_name, relations in related_info.items():
            if entity_name not in kg_info:
                kg_info[entity_name] = []
            for relation in relations:
                kg_info[entity_name].append((relation['relation'], relation['tail']))
    return kg_info

def format_kg(kg_info):
    formatted_kg = []
    for entity, relations in kg_info.items():
        formatted_kg.append(f"{entity}:")
        for relation, tail in relations:
            formatted_kg.append(f"- {relation}: {tail}")
    return "\n".join(formatted_kg)
