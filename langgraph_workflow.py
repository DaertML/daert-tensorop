import os
import subprocess
import matplotlib.pyplot as plt
import networkx as nx
from agents.hardware_factors_agent import HardwareFactorsAgent
from agents.hardware_optimization_hints_agent import HardwareOptimizationHintsAgent
from agents.sketch_kernel_generation_agent import SketchKernelGenerationAgent
from agents.kernel_generation_agent import KernelGenerationAgent
from langgraph.graph import Graph
from langgraph.graph import StateGraph  # Changed from Graph to StateGraph
from ollama import chat
from pydantic import BaseModel, Field  # Added for state modeling
from embed import *

MODEL = 'qwen2.5-coder:32b'

def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

def generate_commit_message(file_path):
    """Generate a commit message for the given file."""
    prompt = f"Please provide a concise commit message (about 10 words) for the changes made to {file_path}:\n{read_file(file_path)}"
    response = chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    content = response['message']['content']
    return content.strip()

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def create_file(file_path, content):
    """Create a file with the given content."""
    with open(file_path, 'w') as f:
        f.write(content)

def read_file(path):
    """Read the contents of a file."""
    with open(path, 'r') as f:
        return f.read().strip()

def edit_file(path, new_content):
    """Edit the contents of a file."""
    with open(path, 'w') as f:
        f.write(new_content)

def generate(msg):
    start_marker = "### START ###"
    end_marker = "### END ###"
    prompt_with_markers = f"{msg}\nPlease provide your response between the following markers:\n{start_marker}\nand\n{end_marker}"
    
    response = chat(model=MODEL, messages=[
        {
            'role': 'user',
            'content': prompt_with_markers,
        },
    ])
    content = response['message']['content']
    
    # Extract content between start and end markers
    start_index = content.find(start_marker) + len(start_marker)
    end_index = content.find(end_marker, start_index)
    return content[start_index:end_index].strip()

def langgraph_workflow(experiment_dir):
    # Create experiment directory if it does not exist
    create_directory(experiment_dir)
    # Define state to accumulate outputs
    class WorkflowState(BaseModel):
        hw_factors: str = Field(default=None)
        hw_optimization_hints: str = Field(default=None)
        sketch_kernel: str = Field(default=None)
        final_kernel: str = Field(default=None)
        context: str = Field(default=None)
        auto_tuned_kernel: str = Field(default=None)
    
    state = WorkflowState()

    # Define the workflow using LangGraph
    graph = StateGraph(WorkflowState)

    # Step 1: Read prompts from files
    hw_factors_prompt = load_prompt("prompts/hwfactors.txt")
    hw_opt_hints_prompt = load_prompt("prompts/hwopthint.txt")
    sketch_gen_prompt = load_prompt("prompts/sketchgen.txt")
    kernel_gen_prompt = load_prompt("prompts/kernelgen.txt")

    # Initialize agents
    hw_factors_agent = HardwareFactorsAgent()
    hw_opt_hints_agent = HardwareOptimizationHintsAgent()
    sketch_kernel_gen_agent = SketchKernelGenerationAgent()
    kernel_gen_agent = KernelGenerationAgent()


    # Add RAG context (using input from workflow start)
    input_str = "implement the sum of two vectors"
    hwmanual_context = "\n".join([result['text'] for result in search_indexed_pdfs(input_str, k=10)])
    state.context = hwmanual_context
    hw_factors_prompt = hw_factors_prompt.replace("{context}", state.context)
    hw_opt_hints_prompt = hw_opt_hints_prompt.replace("{context}", state.context)

    # Step 2: Generate hardware factors
    def generate_hw_factors(state):
        state.hw_factors = hw_factors_agent.generate_hardware_factors(hw_factors_prompt)
        return state

    def save_hw_factors(state):
        file_path = os.path.join(experiment_dir, 'results', 'hw_factors', 'output.txt')
        create_directory(os.path.dirname(file_path))
        create_file(file_path, state.hw_factors)
        print("\033[92mGenerated Hardware Factors saved to {}/results/hw_factors/output.txt\033[0m\n".format(experiment_dir) + state.hw_factors)

        # Git operations
        subprocess.run(['git', 'add', file_path])
        commit_message = generate_commit_message(file_path)
        subprocess.run(['git', 'commit', '-m', commit_message])

        return state

    # Step 3: Generate hardware optimization hints
    def generate_optimization_hints(state):
        state.hw_optimization_hints = hw_opt_hints_agent.generate_optimization_hints(
            hw_opt_hints_prompt.replace("{hwfactors}", state.hw_factors)
        )
        return state

    def save_optimization_hints(state):
        file_path = os.path.join(experiment_dir, 'results', 'hw_optimization_hints', 'output.txt')
        create_directory(os.path.dirname(file_path))
        create_file(file_path, state.hw_optimization_hints)
        print("\033[94mGenerated Hardware Optimization Hints saved to {}/results/hw_optimization_hints/output.txt\033[0m\n".format(experiment_dir) + state.hw_optimization_hints)

        # Git operations
        subprocess.run(['git', 'add', file_path])
        commit_message = generate_commit_message(file_path)
        subprocess.run(['git', 'commit', '-m', commit_message])

        return state

    # Step 4: Generate sketch kernel
    def generate_sketch_kernel(state):
        prompt = sketch_gen_prompt.replace("{hwfactors}", state.hw_factors).replace("{hwopthints}", state.hw_optimization_hints)
        state.sketch_kernel = sketch_kernel_gen_agent.generate_sketch_kernel(prompt)
        return state

    def save_sketch_kernel(state):
        file_path = os.path.join(experiment_dir, 'results', 'sketch_kernel', 'output.txt')
        create_directory(os.path.dirname(file_path))
        create_file(file_path, state.sketch_kernel)
        print("\033[95mGenerated Sketch Kernel saved to {}/results/sketch_kernel/output.txt\033[0m\n".format(experiment_dir) + state.sketch_kernel)

        # Git operations
        subprocess.run(['git', 'add', file_path])
        commit_message = generate_commit_message(file_path)
        subprocess.run(['git', 'commit', '-m', commit_message])

        return state

    # Step 5: Generate final kernel
    def generate_final_kernel(state):
        prompt = kernel_gen_prompt.replace("{hwfactors}", state.hw_factors) \
                                 .replace("{hwopthints}", state.hw_optimization_hints) \
                                 .replace("{sketch}", state.sketch_kernel)
        state.final_kernel = kernel_gen_agent.generate_kernel(prompt)
        return state

    def save_final_kernel(state):
        file_path = os.path.join(experiment_dir, 'results', 'final_kernel', 'output.txt')
        create_directory(os.path.dirname(file_path))
        create_file(file_path, state.final_kernel)
        print("\033[96mGenerated Final Kernel saved to {}/results/final_kernel/output.txt\033[0m\n".format(experiment_dir) + state.final_kernel)

        # Git operations
        subprocess.run(['git', 'add', file_path])
        commit_message = generate_commit_message(file_path)
        subprocess.run(['git', 'commit', '-m', commit_message])

        return state

    def auto_tune(state):
        """Perform auto-tuning on the generated sketch and kernel."""
        import random
        import math

        # Initialize MCTS tree with root node representing initial state
        class Node:
            def __init__(self, parent=None, action=None, state=None):
                self.parent = parent
                self.action = action
                self.state = state
                self.visits = 0
                self.value = 0.0
                self.children = []

        # Create initial state with actual parameters from the kernel
        root_state = {
            'kernel': state.final_kernel,
            # Add actual tunable parameters from your kernel here
            'block_size': 256,
            'grid_size': 128,
            'memory_layout': 'row_major'
        }
        
        root_node = Node(state=root_state)

        def ucb_select(node):
            """Select a node using the UCB algorithm."""
            best_child = None
            best_score = float('-inf')
            for child in node.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    score = (child.value / child.visits) + 1.41 * (math.sqrt(math.log(node.visits) / child.visits))
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def expand_node(node):
            """Expand a node by generating new actions using LLMs."""
            # Generate actions based on actual tunable parameters
            actions = []
            
            # Block size adjustments
            for adjustment in [-32, 32]:
                actions.append({
                    'type': 'adjust', 
                    'parameter': 'block_size', 
                    'value': adjustment
                })
            
            # Grid size adjustments
            for adjustment in [-16, 16]:
                actions.append({
                    'type': 'adjust', 
                    'parameter': 'grid_size', 
                    'value': adjustment
                })
                
            # Memory layout changes
            for layout in ['row_major', 'col_major']:
                if node.state['memory_layout'] != layout:
                    actions.append({
                        'type': 'change', 
                        'parameter': 'memory_layout', 
                        'value': layout
                    })
                    
            for action in actions:
                new_state = node.state.copy()
                if action['type'] == 'adjust':
                    # Only adjust if parameter exists
                    if action['parameter'] in new_state:
                        param_value = new_state[action['parameter']]
                        new_state[action['parameter']] = param_value + action['value']
                elif action['type'] == 'change':
                    new_state[action['parameter']] = action['value']
                    
                child_node = Node(parent=node, action=action, state=new_state)
                node.children.append(child_node)

        def simulate(node):
            """Simulate the performance of a node."""
            # Placeholder for simulation logic - should be replaced with actual kernel evaluation
            return random.random()

        def backpropagate(node, reward):
            """Backpropagate the reward through the tree."""
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # MCTS loop
        for _ in range(10):  # Reduced iterations for demo
            selected_node = root_node
            while selected_node.children:  # Selection phase
                selected_node = ucb_select(selected_node)

            expand_node(selected_node)  # Expansion phase

            if selected_node.children:
                child_node = random.choice(selected_node.children)
                reward = simulate(child_node)  # Simulation phase
                backpropagate(child_node, reward)  # Backpropagation phase

        # Select the best node based on visits or value
        best_child = max(root_node.children, key=lambda x: x.visits)
        
        # Update state with the best kernel configuration
        state.auto_tuned_kernel = best_child.state['kernel']
        return state

    def save_auto_tuned_kernel(state):
        file_path = os.path.join(experiment_dir, 'results', 'auto_tuned_kernel', 'output.txt')
        create_directory(os.path.dirname(file_path))
        create_file(file_path, state.auto_tuned_kernel)
        print("\033[96mAuto-Tuned Kernel saved to {}/results/auto_tuned_kernel/output.txt\033[0m\n".format(experiment_dir) + state.auto_tuned_kernel)

        # Git operations
        subprocess.run(['git', 'add', file_path])
        commit_message = generate_commit_message(file_path)
        subprocess.run(['git', 'commit', '-m', commit_message])

        return state

    def plot_mcts_evolution(root_node):
        """Plot the MCTS evolution tree."""
        G = nx.DiGraph()
    
        def add_nodes_edges(node, parent_name=None):
            node_name = f"Node_{id(node)}"
            G.add_node(node_name, label=f"Visits: {node.visits}\nValue: {node.value:.2f}")
            if parent_name:
                G.add_edge(parent_name, node_name)
        
            for child in node.children:
                add_nodes_edges(child, node_name)

        add_nodes_edges(root_node)

        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
    
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        plt.title("MCTS Evolution Tree")
        plt.show()

    graph.add_node("generate_hw_factors", generate_hw_factors)
    graph.add_node("save_hw_factors", save_hw_factors)
    graph.add_node("generate_optimization_hints", generate_optimization_hints)
    graph.add_node("save_optimization_hints", save_optimization_hints)
    graph.add_node("generate_sketch_kernel", generate_sketch_kernel)
    graph.add_node("save_sketch_kernel", save_sketch_kernel)
    graph.add_node("generate_final_kernel", generate_final_kernel)  # Added before edges
    graph.add_node("save_final_kernel", save_final_kernel)
    graph.add_node("auto_tune", auto_tune)
    graph.add_node("save_auto_tuned_kernel", save_auto_tuned_kernel)
    graph.add_node("plot_mcts_evolution", plot_mcts_evolution)

    graph.add_edge("generate_hw_factors", "save_hw_factors")
    graph.add_edge("save_hw_factors", "generate_optimization_hints")
    graph.add_edge("generate_optimization_hints", "save_optimization_hints")
    graph.add_edge("save_optimization_hints", "generate_sketch_kernel")
    graph.add_edge("generate_sketch_kernel", "save_sketch_kernel")
    graph.add_edge("save_sketch_kernel", "generate_final_kernel")  # Now valid
    graph.add_edge("generate_final_kernel", "save_final_kernel")
    graph.add_edge("save_final_kernel", "auto_tune")
    graph.add_edge("auto_tune", "save_auto_tuned_kernel")
    graph.add_edge("save_auto_tuned_kernel", "plot_mcts_evolution")

    graph.set_entry_point("generate_hw_factors")
    #graph.set_finish_point("save_auto_tuned_kernel")
    initial_state = WorkflowState(context=hwmanual_context)
    # Execute the workflow
    app = graph.compile()
    result_state = app.invoke(initial_state)

if __name__ == "__main__":
    experiment_dir = 'experiment_4'  # You can change this to any desired directory name
    langgraph_workflow(experiment_dir)
