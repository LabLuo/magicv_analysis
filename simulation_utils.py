import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
import random
import pyecsim as ecs
import matplotlib.pyplot as plt
import time
import math
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, deque
from highlight_text import fig_text
import string
from matplotlib.colors import to_rgb
import threading
from electrokitty import ElectroKitty
from pywinauto import Application, Desktop, findwindows
from pywinauto.keyboard import send_keys
import mph
import signal
import functools
import cv2
from PIL import Image
import pytesseract
from pywinauto import Application, Desktop, clipboard
import mss
from scipy.integrate import solve_ivp
from copy import deepcopy

def save_simulation_data(param_map: Dict, model_name: str, scan_rates: List[float],
                        all_E: Dict, all_i: Dict, folder_path: str):
    """Save parameter dictionary and E/i data for each scan rate"""
    import os
    os.makedirs(folder_path, exist_ok=True)
    
    # Save parameter map as JSON
    param_file = f"{folder_path}/{model_name}_params.json"
    with open(param_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_ready_params = convert_numpy_types(param_map)
        json.dump(json_ready_params, f, indent=2)
    
    # Save E and i data for each scan rate
    for scan_rate in scan_rates:
        scan_key = f"{scan_rate:.2e}"
        data = pd.DataFrame({
            'E': all_E.get(scan_key, []),
            'i': all_i.get(scan_key, [])
        })
        csv_file = f"{folder_path}/{model_name}_scanrate_{scan_key.replace('.', 'p')}.csv"
        data.to_csv(csv_file, index=False)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj

# Simulation functions
class SynthesisParser:
    def __init__(self, text):
        self.text = text
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.stack = []  # stack of branch start nodes (or None for concurrency marker)
        self.loop_heads = {}  # map (level,num) -> list of head nodes
        self.loop_tails = {}  # map (level,num) -> list of tail nodes
        self.last_node = None
        self.main_chain_nodes = []

    def new_node(self, label):
        node = f"{label}{self.node_counter}"
        self.graph.add_node(node, type=label)
        if self.last_node:
            self.graph.add_edge(self.last_node, node)
        self.last_node = node
        self.node_counter += 1
        self.main_chain_nodes.append(node)
        return node

    def parse(self):
        i = 0
        paren_depth = 0  # track parentheses nesting

        while i < len(self.text):
            ch = self.text[i]

            if ch in ["C", "E"]:
                self.new_node(ch)

            elif ch == "(":
                self.stack.append(self.last_node)
                paren_depth += 1

            elif ch == ")":
                if self.stack:
                    self.last_node = self.stack.pop()
                paren_depth = max(paren_depth - 1, 0)

            elif ch.isdigit():
                num = int(ch)
                level = 0  # plain number = main chain loop
                self._register_loop(level, num, paren_depth)

            elif ch.isalpha() and ch.islower():
                # expect letter+digit like "a3" or "b2"
                level = ord(ch) - ord("a") + 1
                if i + 1 < len(self.text) and self.text[i+1].isdigit():
                    num = int(self.text[i+1])
                    self._register_loop(level, num, paren_depth)
                    i += 1  # consume digit
                else:
                    raise ValueError(f"Expected digit after {ch} at pos {i}")

            elif ch == ".":
                self.stack.append(None)

            i += 1

        # Connect tails -> heads at all abstraction levels
        for key in set(self.loop_heads.keys()).union(self.loop_tails.keys()):
            heads = self.loop_heads.get(key, [])
            tails = self.loop_tails.get(key, [])
            # print(f"My heads are {heads} and my tails are {tails}.")

            # Track abstraction levels and counts
            considerate_counts = {0: 0}  # level -> count, start with main level 0
            abject_count = 0  # total letters encountered
            current_level = 0  # current abstraction level

            # Store which nodes we need to connect
            heads_to_connect = []
            tails_to_connect = []

            i = 0
            while i < len(self.text):
                ch = self.text[i]

                if ch == '(':
                    # Move down one level in abstraction
                    current_level += 1
                    try:
                        considerate_counts[current_level]
                    except:
                        considerate_counts[current_level] = 0  # Initialize count for new level

                elif ch == ')':
                    # Move up one level in abstraction
                    current_level = max(0, current_level - 1)

                elif ch in ["C", "E"]:
                    # Found a letter, increment counts
                    abject_count += 1
                    considerate_counts[current_level] += 1

                    #print(f"My node is {ch} and my abject_count is {abject_count-1}. My considerate_count at level {current_level} is {considerate_counts[current_level]-1}.")

                    # Check if this node matches our current key
                    node_id = f"{ch}{abject_count-1}"  # Nodes are 0-indexed

                    # Check if this node is in heads or tails for our key
                    if node_id in heads or node_id in tails:
                        # Check if the considerate count matches the loop number
                        level, num = key
                        if current_level == level and considerate_counts[current_level]-1 == num:
                            heads_to_connect.append(node_id)
                        else:
                            tails_to_connect.append(node_id)

                elif ch.isdigit() or (ch.islower() and i+1 < len(self.text) and self.text[i+1].isdigit()):
                    # Skip loop indicators as they don't affect the count
                    if ch.islower():
                        i += 1  # Skip the digit after lowercase letter

                i += 1

            # Connect tails to heads
            for tail in tails_to_connect:
                for head in heads_to_connect:
                    if tail != head:  # Don't create self-loops
                        self.graph.add_edge(tail, head)
                        #print(f"Added edge from tail {tail} to head {head} for loop {key}")

        return self.graph

    def _register_loop(self, level, num, paren_depth):
        key = (level, num)

        if paren_depth > level:
            # Inside parentheses → TAIL
            self.loop_tails.setdefault(key, []).append(self.last_node)
            #print(f"Registered {self.last_node} as TAIL of loop {key}")
        else:
            # Outside parentheses → HEAD
            if self.loop_heads.get(key) and not self.loop_tails.get(key):
                self.loop_tails.setdefault(key, []).append(self.last_node)
                #print(f"Reclassified {self.last_node} as TAIL of loop {key}")
            else:
                self.loop_heads.setdefault(key, []).append(self.last_node)
                #print(f"Registered {self.last_node} as HEAD of loop {key}")

    def draw(self):
        # Generate random colors for each node
        node_colors = {}
        uppercase_letters = [char for char in self.text if char.isupper()]

        for i, node in enumerate(self.graph.nodes()):
            # Generate a random color
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            node_colors[node] = color

        # Create color list for nodes in the order of the graph
        color_list = [node_colors[node] for node in self.graph.nodes()]

        pos = nx.planar_layout(self.graph)
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw the graph
        nx.draw_networkx(self.graph, pos, ax=ax, node_size=500,
                         node_color=color_list, font_size=10,
                         edgecolors='black', linewidths=1)

        # Prepare highlight_text parameters
        highlight_textprops = []
        highlighted_parts = []
        used_colors = list(node_colors.values())[:len(uppercase_letters)]
        color_index = 0

        # Wrap uppercase letters in angle brackets for highlight_text
        text_for_highlight = ""
        for char in self.text:
            if char.isupper():
                text_for_highlight += f"<{char}>"
                highlight_textprops.append({
                    "color": used_colors[color_index],
                    "fontweight": "bold"
                })
                color_index += 1
            else:
                text_for_highlight += char

        # Use highlight_text for the title
        fig_text(
            s=text_for_highlight.strip(),
            x=0.5, y=0.945,
            fontsize=20,
            color='black',
            highlight_textprops=highlight_textprops,
            ha='center'
        )

        plt.tight_layout()
        plt.show()

class GraphTraverser:
    def __init__(self, graph):
        """
        Initialize with a graph represented as an adjacency list.
        graph: dict where keys are node names and values are lists of neighbors
        """
        self.graph = graph
        self.visited = set()
        self.access_order = {}
        self.access_counter = 0
        self.distances = {}

    def _calculate_longest_chain_distances(self):
        """Calculate the longest chain distance for each node using topological sort approach"""
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for node in self.graph:
            for neighbor in self.graph[node]:
                in_degree[neighbor] += 1

        # Initialize distances and queue
        distances = {node: 0 for node in self.graph}
        queue = deque()

        # Add nodes with in-degree 0 to queue
        for node in self.graph:
            if in_degree[node] == 0:
                queue.append(node)
                distances[node] = 0

        # Process nodes in topological order
        while queue:
            current = queue.popleft()

            for neighbor in self.graph[current]:
                in_degree[neighbor] -= 1
                distances[neighbor] = max(distances[neighbor], distances[current] + 1)

                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        self.distances = distances

    def _dfs(self, node):
        """DFS traversal that handles cycles"""
        if node in self.visited:
            return

        # Mark as visited and assign access order
        self.visited.add(node)
        self.access_counter += 1
        self.access_order[node] = self.access_counter

        # Sort neighbors by distance to longest chain (descending)
        neighbors = self.graph.get(node, [])
        sorted_neighbors = sorted(neighbors,
                                 key=lambda x: self.distances.get(x, 0),
                                 reverse=True)

        # Visit neighbors in sorted order
        for neighbor in sorted_neighbors:
            self._dfs(neighbor)

    def traverse_and_rename(self):
        """
        Main method to traverse the graph and rename nodes.
        Returns a dictionary mapping original node names to new names.
        """
        # Reset state
        self.visited = set()
        self.access_order = {}
        self.access_counter = 0

        # Calculate longest chain distances for sorting
        self._calculate_longest_chain_distances()

        # Find all nodes (including those that might not be in graph as keys)
        all_nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_nodes.update(neighbors)

        # Start DFS from nodes with no incoming edges (if any)
        start_nodes = [node for node in all_nodes
                      if not any(node in neighbors for neighbors in self.graph.values())]

        # If no obvious start nodes, start from all nodes
        if not start_nodes:
            start_nodes = list(all_nodes)

        # Sort start nodes by distance to longest chain
        start_nodes.sort(key=lambda x: self.distances.get(x, 0), reverse=True)

        # Perform DFS
        for node in start_nodes:
            if node not in self.visited:
                self._dfs(node)

        # Create new node names
        renamed_nodes = {}
        for node in all_nodes:
            if node.startswith(('C', 'E')):
                # Keep initial letter, replace index with access order
                new_name = node[0] + str(self.access_order.get(node, 0))
                renamed_nodes[node] = new_name

        return renamed_nodes, self.access_order

    def get_sorted_graph(self):
        """
        Return the graph with neighbors sorted by distance to longest chain.
        """
        self._calculate_longest_chain_distances()

        sorted_graph = {}
        for node, neighbors in self.graph.items():
            sorted_neighbors = sorted(neighbors,
                                     key=lambda x: self.distances.get(x, 0),
                                     reverse=True)
            sorted_graph[node] = sorted_neighbors

        return sorted_graph
        
class SynthesisUnparser:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def longest_chain(self):
        """
        Find the longest simple path in the directed graph.
        """
        longest_path = []

        # Try starting DFS from every node
        for node in self.graph.nodes():
            path = self._dfs_longest(node, visited=set())
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def _dfs_longest(self, node, visited):
        """
        Depth-first search to find the longest simple path starting from `node`.
        Avoid revisiting nodes to prevent infinite loops in cycles.
        """
        visited = visited | {node}
        longest_subpath = [node]

        for neighbor in self.graph.successors(node):
            if neighbor not in visited:
                subpath = [node] + self._dfs_longest(neighbor, visited)
                if len(subpath) > len(longest_subpath):
                    longest_subpath = subpath

        return longest_subpath

    def find_non_main_paths(self, chain):
        """
        Find branch and skip paths not part of the main chain.
        Returns a list of paths (each path is a list of node IDs).
        """
        chain_set = set(chain)
        non_main_paths = []

        # Index of each node in main chain for quick lookup
        chain_index = {n: i for i, n in enumerate(chain)}

        # Explore out-branches
        for i, n in enumerate(chain):
            for succ in self.graph.successors(n):
                if succ not in chain or chain_index[succ] > i + 1:
                    # Found a branch or skip edge
                    path = [n]
                    visited = {n}
                    cur = succ
                    while cur not in visited:
                        path.append(cur)
                        visited.add(cur)

                        # Stop if cur is on the main chain (merge)
                        if cur in chain_set and cur != n:
                            break

                        succs = list(self.graph.successors(cur))
                        if not succs:
                            break
                        cur = succs[0]  # follow first successor for a simple representative path

                    non_main_paths.append(path)

        return non_main_paths

    def find_lead_ins(self, chain, branches):
        """
        Find lead-in nodes/paths not in main chain or branches.
        """
        used_nodes = set(chain)
        for path in branches:
            used_nodes.update(path)

        lead_in_paths = []
        for node in self.graph.nodes():
            if node not in used_nodes:
                # Reconstruct a path from this node forward until it merges or ends
                path = [node]
                visited = {node}
                cur = node
                while True:
                    succs = list(self.graph.successors(cur))
                    if not succs:
                        break
                    nxt = succs[0]
                    if nxt in visited:
                        break
                    path.append(nxt)
                    visited.add(nxt)
                    if nxt in used_nodes:
                        break
                    cur = nxt
                lead_in_paths.append(path)

        return lead_in_paths

    def sort_branches_by_level(self, chain, branches):
        """
        Assign each branch to a level:
        0 - branches containing only 2 nodes and both are in the main chain
        1 - branches ending in a main chain node
        2+ - branches ending in nodes of the previous level
        """
        chain_set = set(chain)
        node_to_level = {n: 0 for n in chain}  # main chain nodes are level 0
        level_branches = defaultdict(list)

        remaining = branches.copy()
        progress = True

        while remaining and progress:
            progress = False
            next_remaining = []

            for path in remaining:
                end = path[-1]

                if end in chain_set:
                    # Ends in a main chain node -> first order
                    if len(path) == 2 and path[0] in chain_set and path[1] in chain_set:
                        assigned_level = 0
                    else:
                        assigned_level = 1
                elif end in node_to_level:
                    # Ends in another branch node -> one higher than that node
                    assigned_level = node_to_level[end] + 1
                else:
                    # Can't resolve yet
                    next_remaining.append(path)
                    continue

                # Assign branch and all its nodes
                level_branches[assigned_level].append(path)
                for n in path:
                    node_to_level[n] = assigned_level

                progress = True

            remaining = next_remaining

        # If any branches are left unresolved, dump them into level 1
        for path in remaining:
            level_branches[1].append(path)
            for n in path:
                node_to_level[n] = 1

        return dict(level_branches)


    def _deduplicate_strings(self, strs):
        """
        Remove strings that are substrings of another string in the list.
        """
        unique = []
        for s in strs:
            if not any((s != t and s in t) for t in strs):
                unique.append(s)
        return unique

    def trim_overlapping_branches(self, paths):
        """
        For any two or more paths that share a series of 2+ nodes,
        trim all but one to the first matching node of the series.
        """
        # Make a copy to avoid modifying the original list while iterating
        trimmed_paths = [path[:] for path in paths]

        n = len(trimmed_paths)
        for i in range(n):
            for j in range(i + 1, n):
                path_i = trimmed_paths[i]
                path_j = trimmed_paths[j]

                # Find the longest contiguous overlap of length >= 2
                max_overlap_len = min(len(path_i), len(path_j))
                for l in range(2, max_overlap_len + 1):
                    for start_i in range(len(path_i) - l + 1):
                        seq_i = path_i[start_i:start_i + l]
                        for start_j in range(len(path_j) - l + 1):
                            seq_j = path_j[start_j:start_j + l]
                            if seq_i == seq_j:
                                # Trim path_j to the first matching node of the overlap
                                trimmed_paths[j] = path_j[:start_j + 1]
                                trimmed_paths[i] = path_i[:start_j + 1]
                                trimmed_paths.append(path_i[start_j + 1:])
                                break
                        else:
                            continue
                        break

        return trimmed_paths

    def deduplicate_paths(self, paths):
        """
        Remove paths that are subpaths of another path in the list.
        Each path is a list of nodes.
        """
        unique = []

        def is_subpath(short, long):
            """Check if 'short' is a contiguous subpath of 'long'."""
            if len(short) > len(long):
                return False
            for i in range(len(long) - len(short) + 1):
                if long[i:i+len(short)] == short:
                    return True
            return False

        # Remove subpaths
        for path in paths:
            if not any(path != other and is_subpath(path, other) for other in paths):
                unique.append(path)

        # Trim overlapping branches
        trimmed_branches = unique #self.trim_overlapping_branches(unique)

        #print(trimmed_branches)

        return trimmed_branches

    def _final_check(self, chain, branch_paths, lead_in_paths):
        """
        Ensure every edge in the graph is represented in either the main chain,
        a branch, or a lead-in. Returns a list of missing edges (if any).
        """
        accounted_edges = set()

        # Main chain edges
        for i in range(len(chain) - 1):
            accounted_edges.add((chain[i], chain[i+1]))

        # Branch edges
        for path in branch_paths:
            for i in range(len(path) - 1):
                accounted_edges.add((path[i], path[i+1]))

        # Lead-in edges
        for path in lead_in_paths:
            for i in range(len(path) - 1):
                accounted_edges.add((path[i], path[i+1]))

        # Check against graph edges
        failures = []
        for u, v in self.graph.edges():
            if (u, v) not in accounted_edges:
                failures.append((u, v))

        return failures

    def to_string(self):
        """
        Return the string representation of the longest chain,
        where nodes are labeled with "E" or "C".
        """
        chain = self.longest_chain()
        return "".join(str(self.graph.nodes[n].get("label", n)) for n in chain)

    def remove_duplicate_blocks(self, final_list):
        i = 0
        seen = set()
        bad_list = []
        while i < len(final_list):
            #print(i)
            token = final_list[i]

            # If we've already seen this string -> remove its block
            if token in seen and not (token.isdigit() or token in ['(', ')']):
                #print(f"REPEAT! {token}")
                # find nearest opening parenthesis before i
                start = i
                while start >= 0 and final_list[start] != "(":
                    start -= 1
                # find nearest closing parenthesis after i
                end = i
                while end < len(final_list) and final_list[end] != ")":
                    end += 1

                # if both parentheses found, remove block
                if start >= 0 and end < len(final_list):
                    bad_list.append(final_list[start+1:end])
                    del final_list[start:end+1]
                    # restart scan from beginning (since list changed)
                    i = 0
                    seen = set()
                    continue
            else:
                #print(f"Add token {token} to repeats")
                seen.add(token)

            i += 1

        return final_list, bad_list


    def to_string_with_markers(self):
        """
        Like to_string(), but inserts '<' after splits (out-degree > 1)
        and '>' after merges (in-degree > 1).
        Also appends non-main paths and lead-ins for debugging/inspection,
        and runs a final edge accounting check.
        """
        # Rename the nodes to improve consistency
        traverser = GraphTraverser(nx.to_dict_of_lists(self.graph))
        renamed_nodes, access_order = traverser.traverse_and_rename()
        # print(renamed_nodes)
        sorted_graph = traverser.get_sorted_graph()
        self.graph = nx.DiGraph(sorted_graph)
        #nx.draw(self.graph, with_labels=True)

        chain = self.longest_chain()
        parts = []

        # Create a map of each node in the main chain to its index
        main_chain_indices = {}
        for idx, node in enumerate(chain):
            main_chain_indices[node] = idx

        # Collect branches
        non_main_paths = self.find_non_main_paths(chain)
        non_main_paths = self.deduplicate_paths(non_main_paths)
        non_main_strs = [
            "".join(str(self.graph.nodes[x].get("label", x)) for x in path)
            for path in non_main_paths
        ]
        non_main_strs = self._deduplicate_strings(non_main_strs)

        # Collect lead-ins
        lead_in_paths = self.find_lead_ins(chain, non_main_paths)
        lead_in_paths = self.deduplicate_paths(lead_in_paths)
        lead_in_strs = [
            "".join(str(self.graph.nodes[x].get("label", x)) for x in path)
            for path in lead_in_paths
        ]
        lead_in_strs = self._deduplicate_strings(lead_in_strs)

        # Final edge check
        failures = self._final_check(chain, non_main_paths, lead_in_paths)
        failure_labels = [
            [self.graph.nodes[u].get('label', u), self.graph.nodes[v].get('label', v)]
            for u, v in failures
        ]

        # Add in the final edges
        if failure_labels:
            non_main_paths.extend(failure_labels)
        if lead_in_paths:
            non_main_paths.extend(lead_in_paths)

        # Now sort the branches
        branch_levels = self.sort_branches_by_level(chain, non_main_paths)
        branch_strs_by_level = {
            lvl: [ "".join(str(self.graph.nodes[x].get("label", x)) for x in paths)
                  for paths in paths_list]
            for lvl, paths_list in branch_levels.items()
        }

        final_list = chain.copy()

        if branch_levels and 1 in branch_levels:
            for branch in branch_levels[1]:
                branch=branch.copy()
                start_node = branch[0]
                end_node = branch[-1]

                # Spur case
                if start_node in chain and end_node not in chain: #and self.graph.out_degree(end_node) == 0
                    branch.pop(0)

                    spur = ['(']
                    spur.extend(branch)
                    spur.append(')')

                    for spur_idx, spur_element in enumerate(spur):
                        final_list.insert(final_list.index(start_node)+spur_idx+1, spur_element)
                # Loop case
                elif start_node in chain and end_node in chain:
                    branch.pop(0)
                    branch.pop(-1)

                    loop = ['(']
                    loop.extend(branch)
                    loop.append(f"{main_chain_indices[end_node]}")
                    loop.append(')')

                    # Insert head_node index at its position in the final list
                    final_list.insert(final_list.index(end_node)+1, f'{main_chain_indices[end_node]}')

                    for loop_idx, loop_element in enumerate(loop):
                        final_list.insert(final_list.index(start_node)+loop_idx+1, loop_element)

                # Lead-in case
                elif start_node not in chain and end_node in chain and self.graph.in_degree(start_node) == 0:
                    first_segment = ['(']
                    head_node = branch.pop()  # last node in lead_in_path
                    first_segment.extend(branch)
                    first_segment.append(f"{main_chain_indices[head_node]}")
                    first_segment.append(')')

                    # Insert head_node index at its position in the final list
                    final_list.insert(final_list.index(head_node)+1, f'{main_chain_indices[head_node]}')

                    # Prepend the first_segment to the final list
                    first_segment.extend(final_list)
                    final_list = first_segment


        # Remove duplicates
        final_list, bad_branches = self.remove_duplicate_blocks(final_list)
        #print(f"Final list after: {final_list}")

        # Helper: collect consecutive numbers after a node in final_list
        def collect_suffix_numbers(final_list, node):
            nums = set()
            try:
                idx = final_list.index(node)
            except ValueError:
                return nums  # Node not present

            i = idx + 1
            while i < len(final_list) and final_list[i].isdigit():
                nums.add(final_list[i])
                i += 1
            return nums


        # Identify the branches that haven't been covered yet
        nodes_covered = set(final_list)
        branches_remaining = []

        for level, brs in branch_levels.items():
            for branch in brs:
                if len(set(branch).union(nodes_covered)) > len(nodes_covered):
                    branches_remaining.append(branch)

                elif len(branch) == 2:
                    if not (branch[0] in chain and branch[1] in chain):
                        #print(f"{branch} might be a contender.")

                        try:
                            # Find first occurrence of branch[0] in final_list
                            start_idx = final_list.index(branch[0])
                        except ValueError:
                            # If branch[0] isn't present, keep it as a remaining branch
                            branches_remaining.append(branch)
                            continue

                        found = False
                        i = start_idx + 1
                        paren_depth = 0

                        while i < len(final_list):
                            token = final_list[i]

                            # Track parentheses depth
                            if token == "(":
                                paren_depth += 1
                            elif token == ")":
                                paren_depth = max(paren_depth - 1, 0)

                            # Check if [ '(', branch[1] ] occurs
                            if token == "(" and i + 1 < len(final_list) and final_list[i+1] == branch[1]:
                                found = True
                                break

                            # If we encounter a top-level element starting with E or C, stop searching
                            if paren_depth == 0 and (token.startswith("E") or token.startswith("C")):
                                break

                            i += 1

                        # Extra check: overlapping numeric suffixes
                        suffix0 = collect_suffix_numbers(final_list, branch[0])
                        suffix1 = collect_suffix_numbers(final_list, branch[1])
                        overlap = suffix0.intersection(suffix1)

                        # Decision: add only if no disqualifying case found
                        if not found and not overlap:
                            branches_remaining.append(branch)

        #print("Higher-order branches not yet in final list:", branches_remaining)

# ====This is where I should check for duplicate nodes within branches====
        def get_special_index(final_list, repeated_node):
            """
            Get special index: count only elements inside parentheses
            that start with 'C' or 'E'.
            """
            special_index = 0
            i = 0
            while i < len(final_list):
                if final_list[i] == "(" and i + 1 < len(final_list):
                    start_token = final_list[i+1]
                    if start_token.startswith("C") or start_token.startswith("E"):
                        j = i + 1
                        while j < len(final_list) and final_list[j] != ")":
                            if final_list[j] == repeated_node:
                                return special_index
                            special_index += 1
                            j += 1
                i += 1
            return None


        # --- Resolve remaining branches ---
        branches_removed = []
        for branch in branches_remaining[:]:  # copy so we can modify
            repeated_node = None
            for elem in branch[1:]:
                if elem in final_list:
                    repeated_node = elem
                    break

            if not repeated_node:
                continue  # nothing to resolve

            # Lop off the branch starting from repeated_node
            try:
                cut_index = branch.index(repeated_node)
                chopped_branch = branch[1:cut_index]  # skip branch[0]
            except ValueError:
                continue

            # Compute special index
            special_index = get_special_index(final_list, repeated_node)
            if special_index is None:
                continue

            # Build insertion chunk
            insertion = ["("] + chopped_branch + [f"a{special_index}", ")"]

            try:
                # Find index of branch[0] in final_list
                base_idx = final_list.index(branch[0])
            except ValueError:
                continue

            # Insert after branch[0]
            final_list = final_list[:base_idx+1] + insertion + final_list[base_idx+1:]

            # Also insert after the repeated node
            rep_idx = final_list.index(repeated_node)
            final_list.insert(rep_idx+1, f"a{special_index}")

            branches_removed.append(branch)
            print(f"Resolved {branch} → inserted {insertion} with a{special_index}")

        # Remove the now implemented branches from the remaining_branches
        for branch in branches_removed:
            branches_remaining.remove(branch)

        #print("Higher-order branches not yet in final list:", branches_remaining)

        # Step 1: collect first_order_nodes
        first_order_nodes = []
        inside_paren = False
        for elem in final_list:
            if elem == '(':
                inside_paren = True
            elif elem == ')':
                inside_paren = False
            elif inside_paren:
                # Only add if it's a real node (not an index string)
                if elem[0] in ['C', 'E']:
                    first_order_nodes.append(elem)

        first_order_nodes = set(first_order_nodes)

        #print(first_order_nodes)

        # Create a map of each node in the main chain to its index
        first_order_indices = {}
        for idx, node in enumerate(first_order_nodes):
            first_order_indices[node] = f"a{idx}"

        for branch in branches_remaining:
            #print(f"I'm evaluating {branch}")
            branch=branch.copy()
            start_node = branch[0]
            end_node = branch[-1]

            # Spur case
            if start_node in first_order_nodes and end_node not in first_order_nodes:
                #print("Spur")
                branch.pop(0)

                spur = ['(']
                spur.extend(branch)
                spur.append(')')

                for spur_idx, spur_element in enumerate(spur):
                    final_list.insert(final_list.index(start_node)+spur_idx+1, spur_element)
            # Loop case
            elif start_node in first_order_nodes and end_node in first_order_nodes:
                #print("Loop")
                branch.pop(0)
                branch.pop(-1)

                loop = []

                if len(branch) > 2:
                    loop.append('(')
                loop.extend(branch)
                loop.append(f"{first_order_indices[end_node]}")
                if len(branch) > 2:
                    loop.append(')')

                # Insert head_node index at its position in the final list
                final_list.insert(final_list.index(end_node)+1, f'{first_order_indices[end_node]}')

                for loop_idx, loop_element in enumerate(loop):
                    final_list.insert(final_list.index(start_node)+loop_idx+1, loop_element)

            # Lead-in case
            elif start_node not in first_order_nodes and end_node in first_order_nodes:
                #print("Lead-in")
                first_segment = ['(']
                head_node = branch.pop()  # last node in lead_in_path
                first_segment.extend(branch)
                first_segment.append(f"{first_order_indices[head_node]}")
                first_segment.append(')')

                # Insert head_node index at its position in the final list
                final_list.insert(final_list.index(head_node)+1, f'{first_order_indices[head_node]}')

                # Prepend the first_segment to the final list
                first_segment.extend(final_list)
                final_list = first_segment
            else:
                #print(f"This was the problem case: {branch}")
                pass

# ==== Rewrite the a's to accomodate for our insertions ====
        def remap_special_indices(final_list):
            """
            Remap nodes in single parentheses groups to fresh special indices.
            Returns a dict {node: special_index}.
            """
            # Step 1: collect first_order_nodes
            first_order_nodes = []
            inside_paren = False
            for elem in final_list:
                if elem == '(':
                    inside_paren = True
                elif elem == ')':
                    inside_paren = False
                elif inside_paren:
                    # Only add if it's a real node (not an index string)
                    if elem[0] in ['C', 'E']:
                        first_order_nodes.append(elem)

            # Create a map of each node in the main chain to its index
            mapping = {}
            for idx, node in enumerate(first_order_nodes):
                mapping[node] = idx

            return mapping


        def fix_a_markers(final_list, graph):
            """
            Adjust aX markers so they match the head of the connection
            between the nodes preceding them in final_list.
            """
            node_to_index = remap_special_indices(final_list)
            #print(node_to_index)

            i = 0
            while i < len(final_list):
                token = final_list[i]
                if isinstance(token, str) and token.startswith("a"):
                    idx_val = token[1:]

                    # Find preceding node
                    j = i - 1
                    node1 = None
                    while j >= 0:
                        if final_list[j] in node_to_index:
                            node1 = final_list[j]
                            break
                        j -= 1

                    # Find next aX with same index
                    k = i + 1
                    while k < len(final_list):
                        if final_list[k] == token:  # same aX
                            # Find preceding node for this one
                            m = k - 1
                            node2 = None
                            while m >= 0:
                                if final_list[m] in node_to_index:
                                    node2 = final_list[m]
                                    break
                                m -= 1
                            break
                        k += 1
                    else:
                        i += 1
                        continue  # no pair found

                    # If both nodes are found and connected in the graph
                    if node1 and node2 and graph.has_edge(node1, node2):
                        #print(f"Node1 {node1} and Node2 {node2} have made it!")
                        # Use the head of the connection (example: take node1 as head)
                        head_node = node2
                        new_idx = node_to_index[head_node]
                        new_marker = f"a{new_idx}"

                        # Replace both markers
                        final_list[i] = new_marker
                        final_list[k] = new_marker
                    elif node1 and node2 and graph.has_edge(node2, node1):
                        #print(f"Node1 {node1} and Node2 {node2} have made it!")
                        # Use the head of the connection (example: take node1 as head)
                        head_node = node1
                        new_idx = node_to_index[head_node]
                        new_marker = f"a{new_idx}"

                        # Replace both markers
                        final_list[i] = new_marker
                        final_list[k] = new_marker

                i += 1

            return final_list

        # Remap the special indicies
        final_list = fix_a_markers(final_list, self.graph)

        # Add in base-level directions
        if branch_levels and 0 in branch_levels:
            for branch in branch_levels[0]:
                tail = branch[0]
                head = branch[1]

                final_list.insert(final_list.index(tail)+1, main_chain_indices[head])
                final_list.insert(final_list.index(head)+1, main_chain_indices[head])

        # Construct the final string
        debug_string = []
        final_string = []
        for idx, element in enumerate(final_list):
            # Remove duplicate numbers in a series
            try:
                if str(element[0]) in ['a', 'b'] and element==final_list[idx-1]:
                    continue
                int(element)
                if int(element) == int(final_list[idx-1]) :
                    continue
            except:
                pass

            debug_string.append(str(element)[:] if str(element)[0] not in ['a', 'b'] else element)
            final_string.append(str(element)[0] if str(element)[0] not in ['a', 'b'] else element)

        # print(debug_string)
        return "".join(final_string)
        
import networkx as nx
import random

def generate_ce_digraph(num_nodes, ce_ratio, sparsity, num_final_steps=0, random_state=1):
    """
    Generate a directed graph with nodes labeled 'C' or 'E' according to the CE ratio.
    Ensures that `num_final_steps` nodes have no outgoing edges but at least one incoming edge.
    Every other node will have at least one outgoing edge, and additional edges are
    added according to the sparsity parameter. Ensures the graph is weakly connected.

    Parameters:
    - num_nodes: int, total number of nodes
    - ce_ratio: float [0,1], fraction of nodes that are 'C'. The rest are 'E'.
    - sparsity: float [0,1], probability of extra edges
    - num_final_steps: int, number of nodes with out_degree=0 and in_degree>0

    Returns:
    - G: networkx.DiGraph
    """
    random.seed(random_state)

    if num_final_steps >= num_nodes:
        raise ValueError("num_final_steps must be smaller than num_nodes")

    # Determine the number of C and E nodes
    num_c = int(round(num_nodes * ce_ratio))
    num_e = num_nodes - num_c

    # Create node names
    nodes = [f"C{i}" for i in range(num_c)] + [f"E{i}" for i in range(num_e)]
    random.shuffle(nodes)

    # Initialize directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # Select nodes that will be "final steps" (no out-degree)
    final_nodes = random.sample(nodes, num_final_steps)
    remaining_nodes = [n for n in nodes if n not in final_nodes]

    # Ensure every remaining node has at least one outgoing edge
    for node in remaining_nodes:
        possible_targets = [n for n in nodes if n != node]
        target = random.choice(possible_targets)
        G.add_edge(node, target)

    # Add additional edges based on sparsity
    for node in nodes:
        # skip final nodes for outgoing edges
        if node in final_nodes:
            continue
        for target in nodes:
            if node != target and not G.has_edge(node, target):
                if random.random() < sparsity:
                    G.add_edge(node, target)

    # Ensure each final node has at least one incoming edge
    for node in final_nodes:
        if G.in_degree(node) == 0:
            source = random.choice([n for n in remaining_nodes if n != node])
            G.add_edge(source, node)

    # Ensure weak connectivity
    if not nx.is_weakly_connected(G):
        components = list(nx.weakly_connected_components(G))
        for i in range(len(components) - 1):
            node_from = random.choice(list(components[i]))
            node_to = random.choice(list(components[i + 1]))
            G.add_edge(node_from, node_to)

    return G

def check_ce_digraph(graph):
    results = []
    for idx in range(10):
        unparser = SynthesisUnparser(graph)
        text = unparser.to_string_with_markers()
        results.append(text)

        parser = SynthesisParser(text)
        new_graph = parser.parse()  # do not overwrite original graph
        graph = new_graph  # optional, only if you want to update

    if len(set(results[:])) == 1:
        return graph
    return False


def generate_good_ce_digraph(num_nodes, ce_ratio, sparsity, num_final_steps=1, random_state=1):
    """
    Generate a "good" CE digraph that passes the check_ce_digraph validation
    and also ensures the correct number of final-step nodes (out_degree=0, in_degree>0).
    """
    def valid_final_nodes(graph, expected_final):
        final_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0 and graph.in_degree(n) > 0]
        return (len(final_nodes) == expected_final)

    def valid_size_and_shape(graph, num_nodes):
        return len(graph.nodes) == num_nodes and nx.is_weakly_connected(graph)

    graph = generate_ce_digraph(
        num_nodes=num_nodes,
        ce_ratio=ce_ratio,
        sparsity=sparsity,
        num_final_steps=num_final_steps,
        random_state = random_state
    )

    # Regenerate until both checks pass
    offset = 1
    while not check_ce_digraph(graph) and not valid_final_nodes(graph, num_final_steps) and not valid_size_and_shape(graph, num_nodes):
        random_state += offset
        graph = generate_ce_digraph(
            num_nodes=num_nodes,
            ce_ratio=ce_ratio,
            sparsity=sparsity,
            num_final_steps=num_final_steps,
            random_state = random_state
        )

    return graph


import networkx as nx

def insert_intermediates(G, product_proportion=0.3):
    G = G.copy()
    s_counter = 0
    i_counter = 0
    p_counter = 0

    ce_nodes = [n for n in G.nodes if n.startswith(('C', 'E'))]

    # -------------------------
    # Case A: FIX SPLITS (fan-out)
    # -------------------------
    for node in ce_nodes:
        succs = list(G.successors(node))

        if len(succs) > 1:
            i_node = f"I{i_counter}"
            i_counter += 1
            G.add_node(i_node)

            # Rewire node → I
            for succ in succs:
                G.remove_edge(node, succ)
                G.add_edge(i_node, succ)

            G.add_edge(node, i_node)

    # Recompute after rewiring
    ce_nodes = [n for n in G.nodes if n.startswith(('C', 'E'))]

    # -------------------------
    # Case B: FIX MERGES (fan-in)
    # -------------------------
    for node in ce_nodes:
        preds = list(G.predecessors(node))
        preds_no_i = [el for el in list(G.predecessors(node)) if "I" not in el]

        if not preds:
            s_node = f"S{s_counter}"
            s_counter += 1
            G.add_node(s_node)
            G.add_edge(s_node, node)

        elif len(preds_no_i) > 0:
            i_node = f"I{i_counter}"
            i_counter += 1
            G.add_node(i_node)

            for pred in preds_no_i:
                G.remove_edge(pred, node)
                G.add_edge(pred, i_node)

            G.add_edge(i_node, node)

    # -------------------------
    # Case C: ADD PRODUCT NODES
    # -------------------------
    for node in ce_nodes:
        if G.out_degree(node) == 0:
            p_node = f"P{p_counter}"
            p_counter += 1
            G.add_node(p_node)
            G.add_edge(node, p_node)

    # -------------------------
    # EC' exception fix
    # -------------------------
    if i_counter > 0 and s_counter == 0 and "I0" in G:
        G = nx.relabel_nodes(G, {"I0": "S0"}, copy=False)

    return G


def draw_special(graph, title_text):
    """
    Draw a directed graph with special rules for node types:
      - 'C' and 'E': random colors
      - 'S': white, larger
      - 'P': red, larger
      - 'I': color averages the colors of predecessors, smaller
    """

    # Assign colors to nodes
    node_colors = {}
    node_sizes = {}
    ce_colors = {}

    # First pass: assign random colors to C/E nodes
    for node in graph.nodes():
        if node.startswith(('C', 'E')):
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            node_colors[node] = color
            node_sizes[node] = 500
            ce_colors[node] = color

    # Second pass: assign colors and sizes to S, P, I nodes
    for node in graph.nodes():
        if node.startswith('S'):
            node_colors[node] = "white"
            node_sizes[node] = 800
        elif node.startswith('P'):
            node_colors[node] = "red"
            node_sizes[node] = 800
        elif node.startswith('I'):
            # Average the RGB of predecessors that are C/E
            preds = [p for p in graph.predecessors(node) if p.startswith(('C', 'E'))]
            if preds:
                rgb_vals = [to_rgb(ce_colors[p]) for p in preds]
                avg_rgb = tuple(sum(x)/len(x) for x in zip(*rgb_vals))
                node_colors[node] = avg_rgb
            else:
                node_colors[node] = "#AAAAAA"  # fallback gray
            node_sizes[node] = 300

    # Create color list in node order
    color_list = [node_colors[n] for n in graph.nodes()]
    size_list = [node_sizes[n] for n in graph.nodes()]

    # Layout and figure
    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots(figsize=(6, 6))

    nx.draw_networkx(
        graph, pos, ax=ax,
        node_color=color_list,
        node_size=size_list,
        edgecolors='black',
        linewidths=1,
        font_size=10
    )

    # Highlight uppercase letters in title
    from highlight_text import fig_text
    highlight_textprops = []
    highlighted_chars = [c for c in title_text if c.isupper()]
    used_colors = list(ce_colors.values())[:len(highlighted_chars)]
    color_index = 0
    display_text = ""
    for char in title_text:
        if char.isupper():
            display_text += f"<{char}>"
            highlight_textprops.append({
                "color": used_colors[color_index],
                "fontweight": "bold"
            })
            color_index += 1
        else:
            display_text += char

    fig_text(
        s=display_text.strip(),
        x=0.5, y=0.945,
        fontsize=20,
        color='black',
        highlight_textprops=highlight_textprops,
        ha='center'
    )

    plt.tight_layout()
    plt.show()

# -----------------------------
# Generate random parameters for each node type
# -----------------------------
def generate_node_parameters(G, window=(-1, 1)):
    """
    Generate random parameters for each node in the graph based on its type.

    Parameters:
        G (nx.DiGraph): The reaction graph
        window (tuple): Potential window for electrochemical steps

    Returns:
        dict: Mapping of node names to their parameters
    """
    param_map = {}

    for node in G.nodes():
        if node.startswith('E'):
            # Electrochemical step
            n = np.random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2])
            redox = "oxidation" #np.random.choice(["reduction", "oxidation"])
            E0 = np.random.uniform(window[0], window[1])
            k0 = 10**np.random.uniform(-10, -1)
            alpha = np.random.normal(0.5, 0.05)
            alpha = max(min(alpha, 0.95), 0.05)
            param_map[node] = {'type': 'E', 'params': (n, redox, E0, k0, alpha)}

        elif node.startswith('C'):
            # Chemical step
            # K = 10**np.random.uniform(-7, 7)
            # kf = 10**np.random.uniform(-7, 7)
            # kb = kf / K
            while True:
                kf = 10**np.random.uniform(-6, 6)
                kb = 10**np.random.uniform(-6, 6)
                if (np.log10(kf)**2 + np.log10(kb)**2) < 36:
                    break
            param_map[node] = {'type': 'C', 'params': (kf, kb)}

        elif node.startswith('S'):
            # Starting material
            init_conc = 1 #10**np.random.uniform(-5,0)
            param_map[node] = {'type': 'S', 'params': (init_conc)}

        elif node.startswith('I'):
            # Intermediate species
            init_conc = 0
            param_map[node] = {'type': 'I', 'params': (init_conc)}

        elif node.startswith('P'):
            # Product
            init_conc = 0
            param_map[node] = {'type': 'P', 'params': (init_conc)}

    return param_map

def derive_units_and_rates(G, conc_unit="mM", time_unit="s"):
    """
    Given a bipartite graph G (reaction nodes 'C*'/'E*' and species 'S*'/'I*'/'P*'), generate
      - units for each chemical kf/kb
      - forward/backward rate expressions (strings)
      - ODEs (d[species]/dt) as symbolic strings (mass-action, stoich=1)
    
    Returns:
      dict with keys:
        - 'reaction_summary': {cnode: {...}}
        - 'species_odes': {species: "d[species]/dt = ..."}
    """
    reactions = {}
    # species list
    species_nodes = [n for n in G.nodes() if n.startswith(('S','I','P'))]
    species_odes = {s: [] for s in species_nodes}  # will accumulate terms
    
    for node in G.nodes():
        if not node.startswith('C'):
            continue
        cnode = node
        # reactants: species that have edges into this C node
        reactants = [n for n in G.predecessors(cnode) if n.startswith(('S','I','P'))]
        # products: species that C node points to
        products = [n for n in G.successors(cnode) if n.startswith(('S','I','P'))]
        
        m_fwd = max(1, len(reactants))  # forward molecularity (>=1)
        m_rev = max(1, len(products))   # reverse molecularity (>=1)
        
        # rate expressions (mass-action)
        if reactants:
            fwd_factor = " * ".join([f"c{r}" for r in reactants])
        else:
            fwd_factor = "1"  # unimolecular/zeroth-placeholder (rare)
        if products:
            rev_factor = " * ".join([f"c{p}" for p in products])
        else:
            rev_factor = "1"
        
        kf_name = f"kf_{cnode}"
        kb_name = f"kb_{cnode}"
        fwd_expr = f"{kf_name} * {fwd_factor}"
        rev_expr = f"{kb_name} * {rev_factor}"
        
        # units using formula: [k] = conc^(1-m) * time^-1
        def k_units(m):
            power = 1 - m
            if power == 0:
                conc_part = ""
            elif power == 1:
                conc_part = f"[{conc_unit}]"
            else:
                conc_part = f"[{conc_unit}^{abs(power)}]"
                if power < 0:
                    conc_part = f"[{conc_unit}^({power})]"
            if power == 0:
                return f"[1/{time_unit}]"
            else:
                return f"[{conc_unit}^({1-m}) / {time_unit}]"
        
        kf_units = k_units(m_fwd)
        kb_units = k_units(m_rev)
        
        # record reaction
        reactions[cnode] = {
            "reactants": reactants,
            "products": products,
            "molecularity_forward": m_fwd,
            "molecularity_reverse": m_rev,
            "kf_name": kf_name,
            "kb_name": kb_name,
            "kf_units": kf_units,
            "kb_units": kb_units,
            "forward_rate_expr": fwd_expr,
            "reverse_rate_expr": rev_expr
        }
        
        # Build ODE contributions (stoich = 1 for each species by default)
        # Forward: consumes reactants, produces products
        for r in reactants:
            species_odes[r].append(f"-({fwd_expr})")
        for p in products:
            species_odes[p].append(f"+({fwd_expr})")
        # Reverse: consumes products, produces reactants
        for p in products:
            species_odes[p].append(f"-({rev_expr})")
        for r in reactants:
            species_odes[r].append(f"+({rev_expr})")
    
    # Combine ODE terms into strings
    species_ode_strings = {}
    for s, terms in species_odes.items():
        if not terms:
            expr = "0"
        else:
            expr = " ".join(terms)
            # simplify common "+(-...)" etc not attempted here; this is a readable linear combination
        species_ode_strings[s] = f"d[{s}]/dt = {expr}"
    
    return {
        "reaction_summary": reactions,
        "species_odes": species_ode_strings
    }

def compute_preequilibrium(param_map, G):
    """
    Updates species concentrations in param_map by solving
    pre-equilibrium within chemical-only subgraphs.
    """
    
    param_map = deepcopy(param_map)

    # Helper: identify node types
    def is_species(n): return n.startswith(("S", "I", "P"))
    def is_chemical(n): return n.startswith("C")
    def is_electro(n): return n.startswith("E")

    # Build chemical-only graph
    H = nx.DiGraph()

    for n in G.nodes:
        if is_species(n) or is_chemical(n):
            H.add_node(n)

    for u, v in G.edges:
        if u in H and v in H:
            H.add_edge(u, v)

    # Remove E nodes entirely
    for n in list(H.nodes):
        if is_electro(n):
            H.remove_node(n)

    # Find chemical subgraphs
    components = list(nx.connected_components(H.to_undirected()))

    for comp in components:
        comp = set(comp)

        species_nodes = [n for n in comp if is_species(n)]
        chem_nodes = [n for n in comp if is_chemical(n)]

        if not chem_nodes:
            continue

        # Check if any species has nonzero concentration
        concs = np.array([param_map[n]["params"] for n in species_nodes])
        if np.all(concs == 0):
            continue  # nothing to equilibrate

        # 3. Build reaction list
        reactions = []  # (A, B, kf, kb)

        for Cnode in chem_nodes:
            kf, kb = param_map[Cnode]["params"]

            preds = [p for p in G.predecessors(Cnode) if is_species(p)]
            succs = [s for s in G.successors(Cnode) if is_species(s)]

            # handle multiple reactants/products (all pairwise)
            for A in preds:
                for B in succs:
                    reactions.append((A, B, kf, kb))

        # Species index map
        idx = {s: i for i, s in enumerate(species_nodes)}

        # Define ODE system
        def odes(t, y):
            dydt = np.zeros_like(y)

            for A, B, kf, kb in reactions:
                a = idx[A]
                b = idx[B]

                rate_f = kf * y[a]
                rate_b = kb * y[b]

                dydt[a] += -rate_f + rate_b
                dydt[b] += +rate_f - rate_b

            return dydt

        # Integrate to steady state
        y0 = np.array([param_map[s]["params"] for s in species_nodes])

        # If we throw an error, just return the original
        try:
            sol = solve_ivp(
                odes,
                (0, 1000),          # long time window
                y0,
                method="LSODA",
                rtol=1e-9,
                atol=1e-12
            )
        except:
            return param_map

        y_ss = sol.y[:, -1]

        # Update param_map
        for s in species_nodes:
            param_map[s]["params"] = float(y_ss[idx[s]])

    return param_map
    
# Run simulation using graph
def run_comsol_simulation(G, param_map, comsol_params=None, scan_rates=None):

    """Run COMSOL simulation and return dictionary of results for each scan rate"""
    try:
        # Start COMSOL client + model
        client = mph.start(cores=16)
        model = client.create("mechanism")
    
        # Default parameters if not provided
        if comsol_params is None:
            comsol_params = {
                'startPotential': -1.0,
                'numCycles': 1,
                'vertexPotential1': 1.0,
                'vertexPotential2': -1.0,
                'endPotential': -1.0,
                'electrodeRadius': 1.0,  # mm
            }
        
        # Get the direction of the scan
        start_potential = comsol_params.get('startPotential', -1.0)
        num_cycles = int(comsol_params.get('numCycles', 1)) - 1    # Subtract 1 to make the results more logical
        vertex_potential_1 = comsol_params.get('vertexPotential1', 1.0)
        vertex_potential_2 = comsol_params.get('vertexPotential2', -1.0)
        end_potential = comsol_params.get('endPotential', -1.0)
        
        # Convert electrode radius from mm to m
        electrode_radius_mm = comsol_params.get('electrodeRadius', 1.0)
        electrode_radius = electrode_radius_mm * 1e-3  # Convert mm to m
        electrode_area = math.pi * electrode_radius**2
        
        # Use provided scan rates or create default
        if scan_rates is None:
            # Default scan rates if not provided
            start_scan_rate = comsol_params.get('startScanRate', 0.0001)
            end_scan_rate = comsol_params.get('endScanRate', 10000.0)
            scan_rate_count = comsol_params.get('scanRateCount', 9)
            
            # Create logarithmic range of scan rates
            if scan_rate_count > 1:
                scan_rates = np.logspace(
                    np.log10(start_scan_rate), 
                    np.log10(end_scan_rate), 
                    scan_rate_count
                )
            else:
                scan_rates = [start_scan_rate]

        # Store results for each scan rate
        results = {}
        
        # Treat the graph as a fleshed out graph
        updated_graph = G
        
        # Derive chemical step information
        chem_step_info = derive_units_and_rates(G)
        
        # Rename the parameters and create a call for them
        parameters = model/'parameters'
        (parameters/'Parameters 1').rename('parameters')

        # Create a components and create a call for them
        components = model/'component'
        components.create(True, name='component')

        # Create the model geometry and create a call for it
        geometries = model/'geometries'
        geometry = geometries.create(1, name='geometry')

        # Create the electroanalysis physics model and create a call for it
        physics = model/'physics'
        echem = physics.create('Electroanalysis',geometry, name='echem')

        # Define the lengths of the simulation and diffusion layers we consider
        sim_len = 1e-2 #in meters, so 1 cm
        diff_len = 1e-6 #in meters, so 1 um, this defines the fine meshed regions

        # Parameterize the simulation length
        model.parameter('sim_len', f'{sim_len}[m]') #initconc
        model.description('sim_len', f'Region of solution we consider')

        # Create the diffusion layer
        diff_layer = geometry.create('Interval', name='diff_layer')
        diff_layer.property('lensource','table')
        diff_layer.property('coordvec', np.array([0,diff_len]))

        # Create the bulk layer
        bulk_layer = geometry.create('Interval', name='bulk_layer')
        bulk_layer.property('lensource','table')
        bulk_layer.property('coordvec', np.array([diff_len,sim_len]))

        # Merge the geometries
        model.build(geometry)
        full_domain = model/'mechanism'/'geometries'/'geometry'/'Form Union'

        # Create the concentration parameters
        conc_params = []
        for node, info in param_map.items():
            if info['type'] in ['S', 'I', 'P']:  # species that have concentrations
                init_conc = info['params']  # this is just a scalar for S, I, P
                param_name = f"{node}_init_conc"
                conc_params.append(param_name)

                model.parameter(param_name, f"{init_conc}[mM]")
                model.description(param_name, f"Initial concentration of {node}")
                
        # Store chemical rate constants
        rate_params = []
        for node, info in param_map.items():
            if info['type'] == 'C':  # chemical step
                detailed_info = chem_step_info["reaction_summary"][node]
                kf, kb = info['params']
                
                rate_params.extend([detailed_info["kf_name"], detailed_info["kb_name"]])

                model.parameter(detailed_info["kf_name"], f"{kf}{detailed_info['kf_units']}")   # unit placeholder, adjusted below
                model.parameter(detailed_info["kb_name"], f"{kb}{detailed_info['kb_units']}")   # unit placeholder
                model.description(detailed_info["kf_name"], f"Forward rate constant for {node}")
                model.description(detailed_info["kb_name"], f"Backward rate constant for {node}")
                
        # Store the radius as a parameter (using the user-provided value)
        model.parameter('radius', f'{electrode_radius}[m]')
        model.description('radius', f'Electrode radius: {electrode_radius_mm} mm')

        # Add a circular electrode with that radius (set cross sectional area of electrode based on parameter 'radius')
        echem.java.prop("ac").set("ac", "3.1415*(radius*radius)")

        # Extract SIP nodes in a consistent order
        sip_nodes = [node for node, info in param_map.items() if info['type'] in ['S', 'I', 'P']]

        # Define all concentration in echem module
        for n, node_name in enumerate(sip_nodes):
            echem.java.field('concentration').component(n+1, f"c{node_name}")

        # Define initial concentrations in solution and at semi-infinite boundary
        (echem/'Initial Values 1').property('initc', conc_params)
        echem.create('OpenBoundary', 0, name='Open Boundary')
        (echem/'Open Boundary').java.selection().set(3)
        (echem/'Open Boundary').property('c0', conc_params)

        # Create the Reactions feature in the Electrochemistry interface
        echem.create('Reactions', 1, name='Reactions')

        # Use the generated ODEs to create and store the chemical reactions
        species_odes = chem_step_info["species_odes"]

        # Assign each species' ODE as R_species
        for species, ode_expr in species_odes.items():
            # strip off the "d[species]/dt =" part if needed
            if "=" in ode_expr:
                rhs = ode_expr.split("=", 1)[1].strip()
            else:
                rhs = ode_expr.strip()
            
            (echem/'Reactions').property(f'R_c{species}', rhs)

        # Set where this reaction will be occuring
        (echem/'Reactions').java.selection().set(1, 2)

        # Make electrode surface, assign it to a surface, and check its properties
        echem.create('ElectrodeSurface', 0, name='Electrode')
        (echem/'Electrode').java.selection().set(1)
        (echem/'Electrode').properties()

        # Build index mapping for SIP nodes
        sip_index = {node: i for i, node in enumerate(sip_nodes)}

        # Add in the electrochemical reactions
        first = True
        for n, node in enumerate(G.nodes()):
            params = param_map[node]

            if params['type'] == 'E':
                # Extract electrochemical parameters
                n_e, redox, E0, k0, alpha = params['params']

                # Create new electrode reaction in COMSOL
                if first:
                    (echem/'Electrode'/'Electrode Reaction 1').rename(f'Electrode Reaction {node}')
                    first = False
                else:
                    (echem/'Electrode').create('ElectrodeReaction', 0, name=f'Electrode Reaction {node}')
                (echem/'Electrode'/f'Electrode Reaction {node}').property('nm', f'{n_e}')
                (echem/'Electrode'/f'Electrode Reaction {node}').property('Eeq', f'{E0}[V]')
                (echem/'Electrode'/f'Electrode Reaction {node}').property('k0', f'{k0*electrode_area}[m/s]')
                (echem/'Electrode'/f'Electrode Reaction {node}').property('alphac', f'{alpha*n_e}')

                # Build stoichiometry vector (only SIP species get entries)
                stoich = np.zeros(len(sip_nodes))

                # Reactants = predecessors, products = successors
                for pred in updated_graph.predecessors(node):
                    if pred in sip_index:
                        stoich[sip_index[pred]] -= -1 if redox=='oxidation' else 1
                for succ in updated_graph.successors(node):
                    if succ in sip_index:
                        stoich[sip_index[succ]] += -1 if redox=='oxidation' else 1

                (echem/'Electrode'/f'Electrode Reaction {node}').property('Vi0', stoich)
        
        # Set up the CV conditions
        (echem/'Electrode').property("BoundaryCondition", "CyclicVoltammetry")
        (echem/'Electrode').property("EnableStartPotential", "1")
        (echem/'Electrode').property("Estart", f"{start_potential}[V]")
        (echem/'Electrode').property("ncycle", f"{num_cycles}")
        (echem/'Electrode').property("Evertex1", f"{vertex_potential_1}[V]")
        (echem/'Electrode').property("Evertex2", f"{vertex_potential_2}[V]")
        (echem/'Electrode').property("EnableEndPotential", "1")
        (echem/'Electrode').property("Eend", f"{end_potential}[V]")

        (echem/'Electrode').property("sweeprate", "scan_rate")
        (echem/'Electrode').property("smoothingfactor", "1e-4")
        
        # Make mesh
        (model/'meshes').create(geometry, name='mesh')
        mesh = (model/'meshes'/'mesh')

        # Make diffusion layer mesh
        mesh.create('Edge',name='Edge1')
        (mesh/'Edge1').java.selection().set(1)
        (mesh/'Edge1').create('Distribution',name='Diff')
        (mesh/'Edge1'/'Diff').property('numelem','200')

        # Make bulk layer mesh
        mesh.create('Edge',name='Edge2')
        (mesh/'Edge2').java.selection().set(2)
        (mesh/'Edge2').create('Size',name='Bulk')
        (mesh/'Edge2'/'Bulk').property('custom','on')
        (mesh/'Edge2'/'Bulk').property('hmaxactive','on')
        (mesh/'Edge2'/'Bulk').property('hgradactive','on')
        (mesh/'Edge2'/'Bulk').property('hgrad', 1.05)
        (mesh/'Edge2'/'Bulk').property('hmax', 0.001)
        (mesh/'Edge2'/'Bulk').property('hnarrowactive','off')

        # Loop over different scan rates
        for idx, scan_rate in enumerate(scan_rates):
            print(f"Running simulation for scan rate: {scan_rate} V/s")
            try:
                # Set the scan rate for the simulation
                model.parameter("scan_rate", f"{scan_rate} [V/s]")
                model.description("scan_rate", f"CV scan rate: {scan_rate:.2e} [V/s]")

                # Create the study and examine the model tree
                (model/'studies').create(name=f'CV_{idx}')
                (model/'studies'/f'CV_{idx}').create('CyclicVoltammetry')
                
                # Save and build the model, mesh and solve
                model.build()
                model.mesh('mesh')
                model.solve()

                # Evaluate the results (Potential vs. Current)
                potential = model.evaluate('root.comp1.elan.phis_es1')
                current = model.evaluate('elan.Itot_es1') / electrode_area
                t = model.evaluate("t")

                # Get the amount of distance our scan procedure travels
                total_travel = abs(end_potential - vertex_potential_2) + abs(vertex_potential_2-vertex_potential_1) + abs(vertex_potential_1 - start_potential)
                time_expected = (num_cycles + 1) * total_travel / scan_rate

                # print(f"I expect {time_expected} seconds, and got {max(t)} s. My time array is {len(t)} units long.")

                # If the time-stepping is too small or ran an unreasonable amount of time, fix it
                if len(t) < 100 or (abs(max(t) - time_expected)/time_expected) > 0.01:
                    # Drastically decrease the maximum time step
                    (model/'solutions'/'Solution 1'/'Time-Dependent Solver 1').property("maxstepbdf", f"{time_expected/1000}")

                    # Correct the number of time steps
                    (model/'solutions'/'Solution 1'/'Time-Dependent Solver 1').property("tlist", f"0. {time_expected}") 

                    model.build()
                    model.mesh("mesh")
                    model.solve()

                    # Overwrite results
                    potential = model.evaluate('root.comp1.elan.phis_es1')
                    current = model.evaluate('elan.Itot_es1') / electrode_area
                    t = model.evaluate("t")

                # Store results for this scan rate
                if len(potential) >= 2 and len(current) >= 2:
                    # Store as lists (or numpy arrays) in the results dictionary
                    results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                        'scan_rate': scan_rate,
                        'potential': potential.tolist() if hasattr(potential, 'tolist') else list(potential),
                        'current': current.tolist() if hasattr(current, 'tolist') else list(current),
                        'time': t.tolist() if hasattr(t, 'tolist') else list(t),
                        'electrode_area': electrode_area,
                        'electrode_radius_mm': electrode_radius_mm
                    }
                
                # Clean up for next iteration
                model.clear()
                model.reset()
                
            except Exception as e:
                print(f"Error with scan rate {scan_rate}: {e}")
                # Store error information for this scan rate
                results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                    'scan_rate': scan_rate,
                    'error': str(e),
                    'potential': [],
                    'current': [],
                    'time': []
                }
                continue

        client.remove(model)
        del(model)
            
        return results
        
    except Exception as e:
        print(f"COMSOL simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error results
        error_results = {
            'error': str(e),
            'scan_rates': scan_rates if scan_rates else []
        }
        return error_results
    
    finally:
        try:
            if 'client' in locals():
                client.disconnect()
        except:
            pass

def run_ecsim_simulation(G, param_map, comsol_params=None, scan_rates=None):
    """
    Run ECSim simulation and return dictionary of results for each scan rate
    Mirrors the interface of run_comsol_simulation
    """
    try:
        # Default parameters if not provided
        if comsol_params is None:
            comsol_params = {
                'startPotential': -1.0,
                'numCycles': 1,
                'vertexPotential1': 1.0,
                'vertexPotential2': -1.0,
                'endPotential': -1.0,
                'electrodeRadius': 1.0,  # mm
                'startScanRate': 0.0001,  # V/s
                'endScanRate': 100000.0,  # V/s  
                'scanRateCount': 5,
            }

        # Get the direction of the scan
        start_potential = comsol_params.get('startPotential', -1.0)
        num_cycles = int(comsol_params.get('numCycles', 1))
        vertex_potential_1 = comsol_params.get('vertexPotential1', 1.0)
        vertex_potential_2 = comsol_params.get('vertexPotential2', -1.0)
        end_potential = comsol_params.get('endPotential', -1.0)
        
        # Convert electrode radius from mm to m
        electrode_radius_mm = comsol_params.get('electrodeRadius', 1.0)
        electrode_radius = electrode_radius_mm * 1e-3  # Convert mm to m
        electrode_area = math.pi * electrode_radius**2
        
        # Use provided scan rates or create default
        if scan_rates is None:
            # Default scan rates if not provided
            start_scan_rate = comsol_params.get('startScanRate', 0.0001)  # V/s
            end_scan_rate = comsol_params.get('endScanRate', 10000.0)    # V/s
            scan_rate_count = comsol_params.get('scanRateCount', 9)
            
            # Create logarithmic range of scan rates
            if scan_rate_count > 1:
                scan_rates = np.logspace(
                    np.log10(start_scan_rate), 
                    np.log10(end_scan_rate), 
                    scan_rate_count
                )
            else:
                scan_rates = [start_scan_rate]

        # Store results for each scan rate
        results = {}
        
        # Loop over different scan rates
        for idx, scan_rate in enumerate(scan_rates):
            print(f"Running ECSim simulation for scan rate: {scan_rate} V/s")
            try:
                # Initialize simulation for this scan rate
                sim = ecs.Simulation(True)
                
                # Create species mapping and find starting species
                species_dict = {}
                node_to_concentration = {}

                # First pass: create all species objects
                for node in G.nodes():
                    if node.startswith(('S', 'I', 'P')):  # Species nodes
                        params = param_map.get(node, {})
                        if params.get('type') in ['S', 'I', 'P']:  # Use the param_map type
                            # Starting species
                            init_conc = params['params']
                            species_dict[node] = ecs.Species(node, init_conc, 1.0e-9)
                            node_to_concentration[node] = init_conc

                # Second pass: create reaction objects
                reactions = []
                for node in G.nodes():
                    params = param_map.get(node, {})
                    
                    if params.get('type') == 'E' or params.get('type') == 'C':  # Reaction nodes
                        print("I ENTERED THE IF STATEMENT")
                        # Get reactants and products from graph edges
                        r_nodes = []
                        p_nodes = []
                        reactants = []
                        products = []

                        # Find incoming edges (reactants)
                        for source in G.predecessors(node):
                            if source in species_dict:
                                r_nodes.append(source)
                                reactants.append(species_dict[source])

                        # Find outgoing edges (products)
                        for target in G.successors(node):
                            if target in species_dict:
                                p_nodes.append(target)
                                products.append(species_dict[target])

                        # Skip reactions without proper connections
                        if len(reactants) == 0 or len(products) == 0:
                            print(f"Skipping reaction {node} with no proper connections")
                            continue

                        # Create reaction based on type
                        if params.get('type') == 'E':
                            # Electrochemical reaction
                            n, redox, E0, k0, alpha = params['params']

                            # Determine direction based on redox type
                            if redox == "oxidation" or redox == 1:
                                # For oxidation: products[0] is reduced, reactants[0] is oxidized
                                reaction = ecs.Redox(products[0], reactants[0], int(n), E0, k0, alpha).enable()
                            else: # reduction
                                # Default assumption for reduction
                                reaction = ecs.Redox(reactants[0], products[0], int(n), E0, k0, alpha).enable()
                            
                            reactions.append(reaction)
                            sim.sys.addRedox(reaction)

                        elif params.get('type') == 'C':
                            # Chemical reaction
                            kf, kb = params['params']

                            print(f"Reaction node is {r_nodes[0]} at {reactants[0]}")
                            print(f"Products node is {p_nodes[0]} at {products[0]}")

                            if len(reactants) == 1 and len(products) == 1:
                                # First order: A ⇌ B
                                reaction = ecs.Reaction(reactants[0], None, products[0], None, kf, kb).enable()
                                print(reaction)
                            elif len(reactants) == 2 and len(products) == 1:
                                # Second order: A + B ⇌ C
                                print("I'm where I thought I was going to be.")
                                reaction = ecs.Reaction(reactants[0], reactants[1], products[0], None, kf, kb).enable()
                            elif len(reactants) == 1 and len(products) == 2:
                                # First order with two products: A ⇌ B + C
                                reaction = ecs.Reaction(reactants[0], None, products[0], products[1], kf, kb).enable()
                            elif len(reactants) == 2 and len(products) == 2:
                                # Second order with two products: A + B ⇌ C + D
                                reaction = ecs.Reaction(reactants[0], reactants[1], products[0], products[1], kf, kb).enable()
                            else:
                                # Generic case
                                raise ValueError("A maximum of 2 reactants and 2 products for chemical steps are allowed by ecsim.")

                            reactions.append(reaction)
                            sim.sys.addReaction(reaction)

                # Set electrode and experimental conditions
                sim.el.disk(electrode_radius)
                
                # Set up CV parameters similar to COMSOL
                # ECSim uses setScanPotentials(start, vertex_list, end)
                # For cyclic voltammetry with multiple cycles
                if num_cycles > 1:
                    # Create vertex list for multiple cycles
                    vertices = []
                    for i in range(num_cycles):
                        vertices.append(vertex_potential_1)
                        vertices.append(vertex_potential_2)
                    
                    # Set scan potentials
                    sim.exper.setScanPotentials(start_potential, vertices, end_potential)
                else:
                    # Single cycle
                    sim.exper.setScanPotentials(start_potential, [vertex_potential_1, vertex_potential_2], end_potential)

                sim.exper.setScanRate(scan_rate)

                # Run simulation
                potential, current = sim.run()
                
                # Convert lists to numpy arrays
                potential = np.array(potential)
                current = np.array(current)
                
                # ECSim returns current density in A/m², convert to total current
                current_total = current
                
                # Calculate time array based on scan parameters
                total_travel = abs(end_potential - vertex_potential_2) + abs(vertex_potential_2-vertex_potential_1) + abs(vertex_potential_1 - start_potential)
                total_time = (num_cycles + 1) * total_travel / scan_rate
                
                # Create time array linearly spaced
                time = np.linspace(0, total_time, len(potential))

                # Store results for this scan rate
                if len(potential) >= 2 and len(current_total) >= 2:
                    # Store as lists in the results dictionary
                    results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                        'scan_rate': scan_rate,
                        'potential': potential.tolist(),
                        'current': current_total.tolist(),
                        'time': time.tolist(),
                        'electrode_area': electrode_area,
                        'electrode_radius_mm': electrode_radius_mm,
                        'num_cycles': num_cycles,
                        'start_potential': start_potential,
                        'vertex_potential_1': vertex_potential_1,
                        'vertex_potential_2': vertex_potential_2,
                        'end_potential': end_potential
                    }
                
            except Exception as e:
                print(f"Error with scan rate {scan_rate}: {e}")
                import traceback
                traceback.print_exc()
                
                # Store error information for this scan rate
                results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                    'scan_rate': scan_rate,
                    'error': str(e),
                    'potential': [],
                    'current': [],
                    'time': []
                }
                continue

        return results
        
    except Exception as e:
        print(f"ECSim simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error results
        error_results = {
            'error': str(e),
            'scan_rates': scan_rates if scan_rates else []
        }
        return error_results
    
def run_electrokitty_simulation(G, param_map, comsol_params=None, scan_rates=None):
    """Run ElectroKitty simulation and return dictionary of results for each scan rate"""
    try:
        # Default parameters if not provided
        if comsol_params is None:
            comsol_params = {
                'startPotential': -1.0,
                'numCycles': 1,
                'vertexPotential1': 1.0,
                'vertexPotential2': -1.0,
                'endPotential': -1.0,
                'electrodeRadius': 1.0,  # mm
                'startScanRate': 0.0001,  # V/s
                'endScanRate': 100000.0,  # V/s  
                'scanRateCount': 5,
                'normalizeCurrent': True,
            }

        # Get the direction of the scan
        start_potential = comsol_params.get('startPotential', -1.0)
        num_cycles = int(comsol_params.get('numCycles', 1))
        vertex_potential_1 = comsol_params.get('vertexPotential1', 1.0)
        vertex_potential_2 = comsol_params.get('vertexPotential2', -1.0)
        end_potential = comsol_params.get('endPotential', -1.0)
        
        # Convert electrode radius from mm to m
        electrode_radius_mm = comsol_params.get('electrodeRadius', 1.0)
        electrode_radius = electrode_radius_mm * 1e-3  # Convert mm to m
        electrode_area = math.pi * electrode_radius**2
        print(f"Electrode area: {electrode_area} m²")
        
        # Use provided scan rates or create default
        if scan_rates is None:
            # Default scan rates if not provided
            start_scan_rate = comsol_params.get('startScanRate', 0.0001)  # V/s
            end_scan_rate = comsol_params.get('endScanRate', 10000.0)    # V/s
            scan_rate_count = comsol_params.get('scanRateCount', 9)
            
            # Create logarithmic range of scan rates
            if scan_rate_count > 1:
                scan_rates = np.logspace(
                    np.log10(start_scan_rate), 
                    np.log10(end_scan_rate), 
                    scan_rate_count
                )
            else:
                scan_rates = [start_scan_rate]

        # Store results for each scan rate
        results = {}
        
        # Extract SIP nodes in a consistent order
        sip_nodes = [node for node, info in param_map.items() if info['type'] in ['S', 'I', 'P']]
        
        # Create a mapping for species indices
        species_map = {s: idx for idx, s in enumerate(sip_nodes)}
        
        # Loop over different scan rates
        for idx, scan_rate in enumerate(scan_rates):
            print(f"Running ElectroKitty simulation for scan rate: {scan_rate} V/s")
            try:
                mech_lines = []
                kinetic_constants = []
                diffusion_constants = []
                initial_dissolved = []
                node_to_init_dissolved_order = {}

                # Set initial concentrations for dissolved species
                for idx2, node in enumerate(sip_nodes):
                    init_conc = param_map[node]['params']
                    initial_dissolved.append(init_conc)

                    # Map node → index correctly
                    node_to_init_dissolved_order[node] = idx2

                    diffusion_constants.append(1.0e-9)  # default diffusion coefficient (in m^2/s)

                # Process electrochemical reactions (E nodes)
                for node in G.nodes():
                    params = param_map[node]

                    if params['type'] == 'E':
                        n_e, redox, E0, k0, alpha = params['params']

                        # Identify reactant and product nodes
                        reactants = [s for s in G.predecessors(node) if s in species_map]
                        products  = [s for s in G.successors(node) if s in species_map]

                        if len(reactants) == 1 and len(products) == 1:
                            r, p = reactants[0], products[0]

                            if redox == 1 or redox == "oxidation":
                                # ElectroKitty oxidation convention
                                mech_lines.append(f"E({n_e}): {p} = {r}")
                            else:  # reduction (default)
                                mech_lines.append(f"E({n_e}): {r} = {p}")

                            kinetic_constants.append([alpha, k0, E0])
                    
                    elif params['type'] == 'C':
                        # Chemical reaction
                        kf, kb = params['params']
                        reactants = [s for s in G.predecessors(node) if s in species_map]
                        products = [s for s in G.successors(node) if s in species_map]
                        
                        if len(reactants) == 0 or len(products) == 0:
                            continue
                        
                        reactant_str = "+".join(reactants)
                        product_str = "+".join(products)
                        mech_lines.append(f"C: {reactant_str} = {product_str}")
                        kinetic_constants.append([kf, kb])
                
                mechanism = "\n".join(mech_lines)

                # Build ordered initial_dissolved based on mechanism use
                initial_dissolved = []
                seen_species = set()

                for line in mech_lines:
                    # Split on ":" then "=" to extract species
                    try:
                        _, expr = line.split(":", 1)
                        left, right = expr.split("=")
                    except ValueError:
                        continue  # skip malformed lines

                    # Split species if multiple (C: A+B = C)
                    left_species = [s.strip() for s in left.split("+")]
                    right_species = [s.strip() for s in right.split("+")]

                    for sp in left_species + right_species:
                        if sp not in seen_species:
                            seen_species.add(sp)

                            # Pull the original initial concentration for that species
                            init_conc = param_map[sp]['params']
                            initial_dissolved.append(init_conc)

                # ElectroKitty setup
                spatial_information = [electrode_radius, 20, 1e-5, 0]  # (grid, points, viscosity, rotation)
                cell_constants = [293.15, 0, 0, electrode_area]  # T[K], Ru, Cdl, area[m^2]
                iso = []  # Isolated species
                
                # For multiple cycles, we need to simulate cycle by cycle
                # First cycle
                Ei, Ef = (start_potential, vertex_potential_1)
                simulation = ElectroKitty(mechanism)
                
                # Unpack CV potentials
                E_start = start_potential
                E_v1    = vertex_potential_1
                E_v2    = vertex_potential_2
                E_end   = end_potential

                # Generate a single linear ramp
                def ramp(Ei, Ef, scan_rate):
                    dt = abs(Ef - Ei) / scan_rate
                    npts = 512
                    t = np.linspace(0, dt, npts)
                    E = np.linspace(Ei, Ef, npts)
                    return t, E

                # Build complete waveform
                t_segments = []
                E_segments = []
                t_total = 0.0

                #   start → v1 → v2 → v1 → v2 → ... → end  (numCycles times)
                for c in range(num_cycles):
                    # Start → vertex 1
                    t1, E1 = ramp(E_start if c == 0 else E_v2, E_v1, scan_rate)
                    t1 = t1 + t_total
                    t_total = t1[-1]
                    t_segments.append(t1);  E_segments.append(E1)

                    # Vertex 1 → vertex 2
                    t2, E2 = ramp(E_v1, E_v2, scan_rate)
                    t2 = t2 + t_total
                    t_total = t2[-1]
                    t_segments.append(t2);  E_segments.append(E2)

                # Final segment: last vertex → endPotential
                t3, E3 = ramp(E_v2, E_end, scan_rate)
                t3 = t3 + t_total
                t_segments.append(t3);  E_segments.append(E3)

                # Concatenate full waveform
                t = np.concatenate(t_segments)
                E = np.concatenate(E_segments)

                simulation.set_data(E, np.zeros(len(E)), t) # Here we call the function and pass it our custom signal
                
                simulation.create_simulation(
                    kinetic_constants,
                    cell_constants,
                    diffusion_constants,
                    iso,
                    spatial_information,
                    [[], initial_dissolved],  # adsorbed, dissolved
                    kinetic_model="BV"
                )
                
                # Simulate first segment
                simulation.simulate()
                potential = np.array(simulation.E_generated)
                current = np.array(simulation.current)
                
                # Store results for this scan rate
                if len(potential) >= 2 and len(current) >= 2:
                    # Store as lists (or numpy arrays) in the results dictionary
                    results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                        'scan_rate': scan_rate,
                        'potential': potential.tolist() if hasattr(potential, 'tolist') else list(potential),
                        'current': current.tolist() if hasattr(current, 'tolist') else list(current),
                        'time': t.tolist() if hasattr(t, 'tolist') else list(t),
                        'electrode_area': electrode_area,
                        'electrode_radius_mm': electrode_radius_mm,
                        'num_cycles': num_cycles,
                        'mechanism': mechanism
                    }
                
            except Exception as e:
                print(f"Error with scan rate {scan_rate}: {e}")
                import traceback
                traceback.print_exc()
                # Store error information for this scan rate
                results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                    'scan_rate': scan_rate,
                    'error': str(e),
                    'potential': [],
                    'current': [],
                    'time': []
                }
                continue

        return results
        
    except Exception as e:
        print(f"ElectroKitty simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error results
        error_results = {
            'error': str(e),
            'scan_rates': scan_rates if scan_rates else []
        }
        return error_results
    
def run_digisim_simulation(G_with_intermediates, param_map, comsol_params=None, scan_rates=None):
    """
    Run an electrochemical simulation using DigiSim based on a reaction graph for multiple scan rates.
    
    Parameters:
    - G_with_intermediates: Reaction graph
    - param_map: Parameter mapping
    - model: Placeholder parameter (for consistency with other simulators)
    - comsol_params: Dictionary with simulation parameters (same format as other simulators)
    - scan_rates: list of scan rates to simulate
    
    Returns:
    - Dictionary of results for each scan rate (matching ECsim/COMSOL format)
    """
    time.sleep(2)
    try:
        # Default parameters if not provided
        if comsol_params is None:
            comsol_params = {
                'startPotential': -1.0,
                'numCycles': 1,
                'vertexPotential1': 1.0,
                'vertexPotential2': -1.0,
                'endPotential': -1.0,
                'electrodeRadius': 1.0,  # mm
                'startScanRate': 0.0001,
                'endScanRate': 10000.0,
                'scanRateCount': 9,
                'normalizeCurrent': True,
            }
        
        # Get simulation parameters
        start_potential = comsol_params.get('startPotential', -1.0)
        num_cycles = int(comsol_params.get('numCycles', 1))
        vertex_potential_1 = comsol_params.get('vertexPotential1', 1.0)
        vertex_potential_2 = comsol_params.get('vertexPotential2', -1.0)
        end_potential = comsol_params.get('endPotential', -1.0)
        
        # Convert electrode radius from mm to m
        electrode_radius_mm = comsol_params.get('electrodeRadius', 1.0)
        electrode_radius = electrode_radius_mm * 1e-3  # Convert mm to m
        electrode_area = math.pi * electrode_radius**2
        
        # Use provided scan rates or create default
        if scan_rates is None:
            # Default scan rates if not provided
            start_scan_rate = comsol_params.get('startScanRate', 0.0001)
            end_scan_rate = comsol_params.get('endScanRate', 10000.0)
            scan_rate_count = comsol_params.get('scanRateCount', 9)
            
            # Create logarithmic range of scan rates
            if scan_rate_count > 1:
                scan_rates = np.logspace(
                    np.log10(start_scan_rate), 
                    np.log10(end_scan_rate), 
                    scan_rate_count
                )
            else:
                scan_rates = [start_scan_rate]

        print("DigiSim scan rates:", scan_rates)
        
        # Default DigiSim path (can be moved to parameters if needed)
        digisim_path = r"C:\DigiSim\DigiSim.exe"
        
        # Store results for each scan rate
        results = {}
        
        # Build reaction strings
        reactions = []
        chemical_reactions = []
        reaction_to_node = {}
        reaction_order = []

        # Lists to collect reactants/products for E and C steps
        e_list = []
        c_list = []
        
        # Final ordered list
        final_ordered_list = []
        seen_elements = set()
        
        for node in G_with_intermediates.nodes():
            reaction_order.append(node)
            c_rs = []
            c_ps = []
        
            if node.startswith('E'):
                n, redox, E0, k0, alpha = param_map[node]['params']
                reactants = [s for s in G_with_intermediates.predecessors(node) if s in G_with_intermediates.nodes()]
                products = [s for s in G_with_intermediates.successors(node) if s in G_with_intermediates.nodes()]
                if len(reactants) == 1 and len(products) == 1:
                    r, p = reactants[0], products[0]
                    if redox == "reduction":
                        reactions.append(f"{r}{{SPACE}}{{+}}{{SPACE}}{n}e{{SPACE}}={{SPACE}}{p}")
                        reaction_to_node[f"{r} + {n}e = {p}"] = node
                        e_list.extend([r] + [p])
                    elif redox == "oxidation":
                        reactions.append(f"{p}{{SPACE}}{{+}}{{SPACE}}{n}e{{SPACE}}={{SPACE}}{r}")
                        reaction_to_node[f"{p} + {n}e = {r}"] = node
                        e_list.extend([p] + [r]) # You have to build backwards
        
            elif node.startswith('C'):
                kf, kb = param_map[node]['params']
                reactants = [s for s in G_with_intermediates.predecessors(node) if s in G_with_intermediates.nodes()]
                products = [s for s in G_with_intermediates.successors(node) if s in G_with_intermediates.nodes()]
                if reactants and products:
                    reactant_str = "{SPACE}{+}{SPACE}".join(reactants)
                    product_str  = "{SPACE}{+}{SPACE}".join(products)
                    reactions.append(f"{reactant_str}{{SPACE}}={{SPACE}}{product_str}")
                    chemical_reactions.append(f"{reactant_str}{{SPACE}}={{SPACE}}{product_str}")
                    reaction_to_node[f"{' + '.join(reactants)} = {' + '.join(products)}"] = node
        
                    c_rs = reactants
                    c_ps = products
        
                    # Append to C list: reactants then products
                    c_list.extend(c_rs + c_ps)
        
        # Build final ordered list without duplicates
        for element in e_list + c_list:  # first E, then C
            if element not in seen_elements:
                seen_elements.add(element)
                final_ordered_list.append(element)
        
        # Use CV parameters from comsol_params
        start_point = start_potential
        erev1 = vertex_potential_1
        erev2 = vertex_potential_2
        end_point = end_potential
        
        # First scan rate: full setup
        first_scan_rate = scan_rates[0]
        
        app = Application(backend="uia").start(digisim_path)
        main_window = app.top_window()
        main_window.set_focus()
        time.sleep(0.1)
        
        # Start new document
        main_window.type_keys("^n")
        time.sleep(0.1)

        # Change to IUPAC standard
        view_button = main_window.child_window(title="Application", auto_id="MenuBar", control_type="MenuBar").child_window(title="View", control_type="MenuItem")
        view_button.click_input()
        view_button.child_window(title="View", control_type="Window").child_window(title="View", control_type="Menu").child_window(title="Preferences...", auto_id="32409", control_type="MenuItem").click_input()
        preference_window = main_window.child_window(title="Preferences", control_type="Window")
        preference_window.child_window(title="IUPAC", auto_id="3003", control_type="RadioButton").click_input()
        preference_window.child_window(title="Close", control_type="Button").click_input()
        
        # Access the toolbar and click Button 15 (Reaction Editor)
        toolbar = main_window.child_window(auto_id="59392", control_type="ToolBar")
        toolbar.children()[11].click_input()
        time.sleep(0.1)
        
        # Access the CV properties / reaction input window
        parameter_window = main_window.child_window(title="CV-Properties", control_type="Window")
        
        # Enter reactions one by one
        for rxn in reactions:
            parameter_window.type_keys(rxn)
            parameter_window.type_keys("{ENTER}")
            time.sleep(0.2)  # short pause to ensure DigiSim registers input

        # Figure out reaction order
        # -----------------------
        # Switch to the chemical parameter tab
        chem_tab = parameter_window.child_window(title="Chemical Parameters", control_type="TabItem")
        chem_tab.select()

        # Get scrollbar and down buttons
        try:
            hetero_scrollbar = parameter_window.child_window(auto_id="3059", control_type="ScrollBar")
            hetero_down_button = hetero_scrollbar.child_window(auto_id="DownButton", control_type="Button")
        except:
            pass

        # Highlight the box of interest
        heterogeneous_box = parameter_window.child_window(auto_id="3054", control_type="Group")
        heterogeneous_box.draw_outline()
        
        # Process the E steps
        num_processed = 0
        extracted_text = []
        for reaction in reactions:
            # Clean up the reaction string and grab the node we are talking about
            reaction = reaction.replace('{SPACE}', ' ').replace('{+}', '+')
            node = reaction_to_node[reaction]
            rxn_type = param_map[node]['type']
            if rxn_type == 'E':
                if num_processed > 2:
                    # Scroll down when needed and get back to the new line
                    hetero_down_button.click_input()

                time.sleep(0.2)

                if int(num_processed) not in [1, 2]:
                    # Take screenshot
                    with mss.mss() as sct:
                        sct.shot(output="screen.png")

                    # Isolate the hetereogenous reactions
                    original_screenshot = Image.open("screen.png")
                    cropped_screenshot = original_screenshot.crop((712, 1080-735, 850, 1080-625))
                    # cropped_screenshot.show()
                    cropped_screenshot.save("cropped_screenshot.png")

                    # Read in the image for cv2 processing
                    image = cv2.imread("cropped_screenshot.png")

                    # Upsample, grey, and thresh out the image
                    scale = 10.0
                    upsampled = cv2.resize(
                        image,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC  # best for upscaling
                    )
                    gray = cv2.cvtColor(upsampled, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

                    # Use pytesseract to extract the text
                    custom_config = r'--psm 6' #r'-c tessedit_char_whitelist=eSIP0123456789=+ --psm 6'
                    data = pytesseract.image_to_string(thresh, lang='eng', config=custom_config)

                    # Store the text
                    extracted_text.append(data)

                # Increment the number of nodes we have processed
                num_processed += 1

        # Build the Digisim-extracted string
        digisim_string = extracted_text[0]
        try:
            for reaction_strings in extracted_text[1:]:
                digisim_string += reaction_strings.split('\n')[-2] + '\n'  # take second-to-last line
        except:
            pass

        # Normalization function
        def normalize(s):
            replacements = {
                'I': '1',
                'l': '1',
                '|': '1',
                '/': '1',
                'T': '1',
                'O': '0',
                's': '+',
                'S': '+',
            }
            for k, v in replacements.items():
                s = s.replace(k, v)
            
            # Special remapping for '+1le=' -> '+1e=' and '+12e=' -> '+2e='
            s = re.sub(r'\+11?e=', '+1e=', s)
            s = re.sub(r'\+12e=', '+2e=', s)
            
            return re.sub(r'\s+', '', s)  # remove all whitespace

        # Extract lines containing '='
        digisim_lines = [line for line in digisim_string.splitlines() if '=' in line]

        print("Digisim lines", digisim_lines)

        # Normalize
        normalized_main = [normalize(line) for line in digisim_lines]
        normalized_reactions = {
            normalize(r.replace("{SPACE}", '').replace('{', '').replace('}', '')): r
            for r in reactions
        }

        # Match in order
        ordered_reactions = []
        for nm in normalized_main:
            for nr, original in normalized_reactions.items():
                if nr == nm:
                    ordered_reactions.append(original)
                    break

        # Get the true node order
        node_order = []
        seen = set()
        for line in ordered_reactions:
            matches = re.findall(r'[ISP]\d+', line)  # capture full I#, S#, P# strings
            for m in matches:
                if m not in seen:
                    node_order.append(m)
                    seen.add(m)

        # Final ordered list
        final_ordered_list = []
        seen_elements = set()
        for element in node_order + c_list:  # first E, then C
            if element not in seen_elements:
                seen_elements.add(element)
                final_ordered_list.append(element)

        order_list = final_ordered_list

        reactions = ordered_reactions + chemical_reactions

        # Access the tab bar at the top
        parameter_window.set_focus()
        tab_control = parameter_window.child_window(auto_id="12320", control_type="Tab")
        
        # Click the 'CV-Parameters' tab
        tab_control.child_window(title="CV-Parameters", control_type="TabItem").select()

        ru = 0
        cdl = 0
        temp = 293.15
        parameter_window.type_keys(f"{start_point}{{ENTER}}{erev1}{{ENTER}}{erev2}{{ENTER}}{first_scan_rate}{{ENTER}}{num_cycles}{{ENTER}}{ru}{{ENTER}}{cdl}{{ENTER}}{temp}{{ENTER}}")
        
        # --- Initial concentrations ---
        # Get visible edit boxes
        def get_visible_edit_boxes(window, base_id, boxes_per_page):
            """Return all visible, enabled Edit boxes in order."""
            visible_boxes = []
            for i in range(boxes_per_page):
                box_id = str(base_id + i)
                try:
                    box = window.child_window(auto_id=box_id, control_type="Edit")
                    if box.is_visible() and box.is_enabled():
                        visible_boxes.append(box)
                except Exception:
                    continue
            return visible_boxes
        
        # Try to find scrollbar
        try:
            scrollbar = parameter_window.child_window(auto_id="3035", control_type="ScrollBar")
            down_button = scrollbar.child_window(auto_id="DownButton", control_type="Button")
            print("Scrollbar found — will scroll as needed.")
            scroll_exists = True
        except findwindows.ElementNotFoundError:
            print("No scrollbar found — all boxes should be visible.")
            scroll_exists = False
        
        base_id = 3120          # First species box (S0)
        boxes_per_page = 3      # Number of visible boxes
        species_index = 0
        
        # First: fill all visible boxes (up to boxes_per_page)
        visible_boxes = get_visible_edit_boxes(parameter_window, base_id, boxes_per_page)
        print(f"Found {len(visible_boxes)} visible boxes")
        
        # Fill initial visible boxes
        for i, box in enumerate(visible_boxes):
            if species_index >= len(order_list):
                break
                
            node = order_list[species_index]
            init_conc = param_map[node]['params'] / 1000 # Convert from [mM] to [M]
            
            try:
                box.set_focus()
                send_keys(f"^a{{BACKSPACE}}{str(init_conc)}{{ENTER}}")  # clear current contents
                print(f"Filled {node} with {init_conc}")
                species_index += 1
            except Exception as e:
                print(f"Could not fill visible box for {node}: {e}")
                species_index += 1
        
        # If there are more species, scroll and replace the last box content
        while scroll_exists and species_index < len(order_list):
            try:
                # Scroll down
                down_button.click_input()
                
                # Get the new set of visible boxes after scrolling
                visible_boxes = get_visible_edit_boxes(parameter_window, base_id, boxes_per_page)
                if not visible_boxes:
                    print("No visible boxes found after scrolling")
                    break
                    
                # Focus on the last visible box
                last_box = visible_boxes[-1]
                last_box.click_input()
                
                # Fill the last box with the next species
                node = order_list[species_index]
                init_conc = param_map[node]['params'] / 1000 # Convert from [mM] to [M]
                
                send_keys("^a{BACKSPACE}")  # clear current contents
                send_keys(str(init_conc))
                send_keys("{ENTER}")
                print(f"Filled {node} with {init_conc} (after scrolling)")
                
                species_index += 1
                
            except Exception as e:
                print(f"Error during scrolling/filling: {e}")
                break
        
        # Add in the area
        area = math.pi * (electrode_radius*100)**2 #rad in m, convert to cm
        parameter_window.type_keys(f"{{ENTER}}^a{{BACKSPACE}}{area}")
        
        # Turn off the pre-equilibrium
        pre_eq_disabled = parameter_window.child_window(auto_id="3034", control_type="RadioButton")
        pre_eq_disabled.select()
        
        # Switch to the chemical parameter tab
        chem_tab = parameter_window.child_window(title="Chemical Parameters", control_type="TabItem")
        chem_tab.select()
        
        try:
            # Define the heterogeneous reaction scrollbar
            hetero_scrollbar = parameter_window.child_window(auto_id="3059", control_type="ScrollBar")
            hetero_down_button = hetero_scrollbar.child_window(auto_id="DownButton", control_type="Button")
        except:
            pass
        
        try:
            # Define the homoeneous reaction scrollbar
            homo_scrollbar = parameter_window.child_window(auto_id="3060", control_type="ScrollBar")
            homo_down_button = homo_scrollbar.child_window(auto_id="DownButton", control_type="Button")
        except:
            pass
        
        # Process the E steps
        num_processed = 0
        for reaction in reactions:
            # Clean up the reaction string and grab the node we are talking about
            reaction = reaction.replace('{SPACE}', ' ').replace('{+}', '+')
            node = reaction_to_node[reaction]
            rxn_type = param_map[node]['type']
            if rxn_type == 'E':
                if num_processed > 2:
                    # Scroll down when needed and get back to the new line
                    hetero_down_button.click_input()
                    parameter_window.type_keys("{ENTER}{ENTER}{ENTER}{ENTER}")
        
                # Grab and type the parameters
                n_e, ro, E0, k0, alpha = param_map[node]['params']
                
                parameter_window.type_keys(E0)
                parameter_window.type_keys("{ENTER}")
        
                parameter_window.type_keys(alpha)
                parameter_window.type_keys("{ENTER}")
        
                parameter_window.type_keys(k0*100) # convert from [m/s] to [cm/s]
                parameter_window.type_keys("{ENTER}")
        
                # Increment the number of nodes we have processed
                num_processed += 1
        
        # Process the C steps
        num_processed = 0
        for reaction in reactions:
            # Clean up the reaction string and grab the node we are talking about
            reaction = reaction.replace('{SPACE}', ' ').replace('{+}', '+')
            node = reaction_to_node[reaction]
            rxn_type = param_map[node]['type']
            if rxn_type == 'C':
                if num_processed > 2:
                    # Scroll down when needed and get back to the new line
                    homo_down_button.click_input()
                    parameter_window.type_keys("{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}{ENTER}")
                    
                # Grab and type the parameters
                kf, kb = param_map[node]['params']
                
                parameter_window.type_keys(kf/kb)
                parameter_window.type_keys("{ENTER}")
        
                parameter_window.type_keys(kf)
                parameter_window.type_keys("{ENTER}")
        
                # Increment the number of nodes we have processed
                num_processed += 1

        # Save the settings
        tab_control.child_window(title="CV-Parameters", control_type="TabItem").select()
        parameter_window.child_window(auto_id="1", control_type="Button").click_input()

        # Function to read DigiSim results file
        def read_cv_data_robust(filename):
            """
            More robust version that handles various data formats.
            """
            potentials = []
            currents = []
            
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    # Skip empty lines and header lines
                    if not line or any(keyword in line for keyword in 
                                      ["source program:", "experimental parameters:", 
                                       "data statistics:", "number of E(V), I(A) couples:"]):
                        continue
                        
                    # Look for data lines (contain comma and scientific notation)
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            # Clean up the values
                            e_str = parts[0].strip()
                            i_str = parts[1].strip().upper()  # Convert to uppercase for E notation
                            
                            # Handle scientific notation (E-035 -> E-35)
                            i_str = re.sub(r'E-0*(\d+)', r'E-\1', i_str)
                            i_str = re.sub(r'E\+0*(\d+)', r'E+\1', i_str)
                            
                            try:
                                e_value = float(e_str)
                                i_value = float(i_str)
                                
                                potentials.append(e_value)
                                currents.append(i_value)
                            except ValueError:
                                continue
            
            return potentials, currents
        
        # Function to run simulation and extract results
        def run_simulation_and_extract():
            # Run the simulation
            menu_bar = main_window.child_window(title="Application", auto_id="MenuBar", control_type="MenuBar")
            run_button = menu_bar.child_window(title="Run", control_type="MenuItem")
            run_button.click_input()
            run_button.child_window(title="Run", control_type="Window").child_window(title="Run", control_type="Menu").child_window(title="Simulation", auto_id="32774", control_type="MenuItem").click_input()
            
            # Export the results
            toolbar.children()[4].click_input()
            main_window.child_window(title="File name:", auto_id="1152", control_type="Edit").type_keys("^a{DELETE}temp_file.use")
            main_window.child_window(title="Export", control_type="Window").child_window(title="Export", auto_id="1", control_type="Button").click_input()

            # Deal with problems whilst exporting
            try:
                main_window.child_window(title="Export", control_type="Window").child_window(title="Confirm Save As", control_type="Window").child_window(title="Yes", auto_id="CommandButton_6", control_type="Button").click_input()
            except:
                print("No save confirmation dialog")
            
            # Extract the results
            return read_cv_data_robust(r"C:\Users\LabLu\COMSOL\Connor\Graphs\temp_file.use")

        # Run simulation for all scan rates
        for scan_rate in scan_rates:
            
            # For first scan rate, use existing setup
            if scan_rate == first_scan_rate:
                E, i = run_simulation_and_extract()
            else:
                # For subsequent scan rates, modify the scan rate
                # Open CV properties again
                toolbar.children()[11].click_input()
                
                # Access the CV parameters tab
                parameter_window = main_window.child_window(title="CV-Properties", control_type="Window")
                tab_control = parameter_window.child_window(auto_id="12320", control_type="Tab")
                
                # Navigate to scan rate field (4th field after start, rev, end points)
                parameter_window.type_keys("{ENTER}{ENTER}{ENTER}")
                
                # Clear and enter new scan rate
                parameter_window.type_keys("^a{DELETE}")
                parameter_window.type_keys(str(scan_rate))
                parameter_window.type_keys("{ENTER}")
                
                # Save settings
                parameter_window.child_window(auto_id="1", control_type="Button").click_input()
                
                # Run simulation and extract results
                E, i = run_simulation_and_extract()
            
            # Store results in dictionary with same format as other simulators
            if len(E) >= 2 and len(i) >= 2:
                results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                    'scan_rate': scan_rate,
                    'potential': E,
                    'current': i,
                    'electrode_area': electrode_area,
                    'electrode_radius_mm': electrode_radius_mm,
                    'num_cycles': num_cycles,
                    'mechanism': "\n".join([r.replace('{SPACE}', ' ').replace('{+}', '+') for r in reactions])
                }
            else:
                results[f"scan_rate_{scan_rate:.2e}_V_s"] = {
                    'scan_rate': scan_rate,
                    'error': 'No valid data extracted',
                    'potential': [],
                    'current': []
                }

        # Close the simulation
        main_window.child_window(title="Close", control_type="Button", found_index=0).click_input()

        # Handle any save dialog that might appear
        try:
            time.sleep(0.5)
            save_dialog = app.top_window()
            save_dialog.child_window(title="No", control_type="Button").click_input()
        except:
            pass  # No save dialog appeared

        app.kill()
        
        return results
        
    except Exception as e:
        print(f"DigiSim simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error results
        error_results = {
            'error': str(e),
            'scan_rates': scan_rates if scan_rates else []
        }
        return error_results