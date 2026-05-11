from simulation_utils import *

import concurrent.futures
import threading
import time
from typing import Dict, List, Any
import copy
import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc

class GeneralSimulationRunner:
    def __init__(self, mechanism: str = "ECE", num_simulations: int = 250, 
                 random_state: int = 60, total_cores: int = 8):
        self.num_simulations = num_simulations
        self.random_state = random_state
        self.total_cores = total_cores
        
        # Core allocation
        self.digisim_workers = 1  # Serial only
        # Leave 1 core for system, distribute remaining among simulators; they run one after the other, so the remainder can be given to each
        self.parallel_workers = total_cores - 1
        
        print(f"Core allocation: DigiSim=1, Parallel simulators={self.parallel_workers} each")
        
        # Create executors
        self.digisim_executor = ThreadPoolExecutor(max_workers=self.digisim_workers)
        self.parallel_executor = ProcessPoolExecutor(max_workers=self.parallel_workers)
        
        # Simulation parameters
        self.scan_rates = np.logspace(-2, 2, 5).tolist()
        self.initial_concentration = 1
        self.electrode_radius = 1.0
        self.num_cycles = 1
        
        # Get the graph
        if mechanism.endswith(".gml"):
            self.mechanism = mechanism[:-4]
            self.G_with_intermediates = nx.read_gml(mechanism)
        else:
            self.mechanism = mechanism
            parser = SynthesisParser(mechanism)
            G = parser.parse()
            parser.draw()
            self.G_with_intermediates = insert_intermediates(G)

        # Create parameter map
        self.base_param_map = generate_node_parameters(self.G_with_intermediates)
        
        # Common comsol_params
        self.comsol_params = {
            'startPotential': 1.0,
            'numCycles': self.num_cycles,
            'vertexPotential1': -1.0,
            'vertexPotential2': 1.0,
            'endPotential': 1.0,
            'electrodeRadius': self.electrode_radius,
            'startScanRate': 0.01,
            'endScanRate': 100.0,
            'scanRateCount': len(self.scan_rates)
        }
        
        # Storage
        self.param_maps = {}
        self.results = {}
        
        # Create directories
        for model_id in range(self.num_simulations):
            folder_name = f"model_{model_id:04d}"
            os.makedirs(f"simulated_data/{self.mechanism}/{folder_name}", exist_ok=True)
            self.results[model_id] = {
                'digisim': None,
                'ecsim': None,
                'electrokitty': None,
                'comsol': None,
                'folder': folder_name,
                'params_saved': False
            }
    
    def generate_all_param_maps(self):
        """Generate all parameter maps"""
        print(f"Generating {self.num_simulations} parameter maps...")
        
        for model_id in range(self.num_simulations):
            random_state = self.random_state + model_id
            param_map = generate_randomized_param_map(self.base_param_map, random_state)
            self.param_maps[model_id] = param_map
            self._save_parameters(model_id, param_map)
            self.results[model_id]['params_saved'] = True
            
            if (model_id + 1) % 50 == 0:
                print(f"Generated {model_id + 1}/{self.num_simulations} parameter maps")
    
    def _save_parameters(self, model_id: int, param_map: Dict):
        """Save parameters for a single model"""
        folder_name = self.results[model_id]['folder']
        param_filename = f"simulated_data/{self.mechanism}/{folder_name}/model_{model_id:04d}_params.json"
        
        with open(param_filename, 'w') as f:
            json_ready = convert_numpy_types(param_map)
            json.dump(json_ready, f, indent=2)
    
    def _save_simulator_results(self, model_id: int, simulator: str, results: Dict):
        """Save results for a single simulator"""
        if results is None:
            print(f"Warning: No results for {simulator} model {model_id}")
            return
        
        print("Results:", results)
            
        folder_name = self.results[model_id]['folder']
        
        for scan_key, data in results.items():
            if 'potential' in data.keys() and 'current' in data.keys():
                df_data = {
                    'E': data['potential'],
                    'i': data['current']
                }
                if 'time' in data:
                    df_data['t'] = data['time']
                
                df = pd.DataFrame(df_data)
                filename_scan_key = str(scan_key).replace('.', 'p').replace('+', '')
                filename = f"simulated_data/{self.mechanism}/{folder_name}/model_{model_id:04d}_{simulator}_scan_{filename_scan_key}.csv"
                df.to_csv(filename, index=False)
        
        print(f"Saved {simulator} results for model {model_id}")
    
    def _handle_simulator_error(self, model_id: int, simulator: str, error: str):
        """Handle errors from simulators"""
        print(f"Error in {simulator} for model {model_id}: {error}")
        folder_name = self.results[model_id]['folder']
        error_filename = f"simulated_data/{self.mechanism}/{folder_name}/model_{model_id:04d}_{simulator}_error.txt"
        with open(error_filename, 'w') as f:
            f.write(f"Error in {simulator}: {error}\n")
        gc.collect()
    
    def _run_model_type(self, simulator: str, executor, model_ids: List[int], 
                        run_func, max_workers: int = None):
        """
        Run a single model type across all models
        
        Args:
            simulator: Name of simulator ('digisim', 'ecsim', etc.)
            executor: ThreadPoolExecutor or ProcessPoolExecutor
            model_ids: List of model IDs to run
            run_func: Function to call for each model
            max_workers: Max concurrent workers (if None, use executor's max_workers)
        """
        print(f"Running {simulator.upper()} for {len(model_ids)} models")
        print(f"Max workers: {max_workers or executor._max_workers}")
        
        futures = {}
        
        # Submit all tasks
        for model_id in model_ids:
            param_map = self.param_maps[model_id]
            
            # Get redox type for comsol_params
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))
            
            if not e_keys:
                print(f"Warning: No E<n> keys for model {model_id}, skipping")
                continue
            
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])
            redox_type = param_map[lowest_e_key]["params"][1]
            
            comsol_params = copy.deepcopy(self.comsol_params)
            if redox_type == "reduction":
                comsol_params.update({
                    'startPotential': 1.0,
                    'vertexPotential1': -1.0,
                    'vertexPotential2': 1.0,
                    'endPotential': 1.0
                })
            elif redox_type == "oxidation":
                comsol_params.update({
                    'startPotential': -1.0,
                    'vertexPotential1': 1.0,
                    'vertexPotential2': -1.0,
                    'endPotential': -1.0
                })
            
            # For DigiSim, we need to handle preequilibrium
            if simulator == 'digisim':
                param_map = compute_preequilibrium(param_map, self.G_with_intermediates)
            
            # Submit the task
            future = executor.submit(
                run_func,
                self.G_with_intermediates, param_map, comsol_params, self.scan_rates
            )
            futures[future] = model_id
        
        # Process results as they complete
        completed = 0
        failed = 0
        
        for future in as_completed(futures):
            model_id = futures[future]
            completed += 1
            
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                self._save_simulator_results(model_id, simulator, result)
                print(f"[{simulator}] Completed {completed}/{len(model_ids)} - Model {model_id}")
            except Exception as e:
                failed += 1
                self._handle_simulator_error(model_id, simulator, str(e))
                print(f"[{simulator}] Failed {completed}/{len(model_ids)} - Model {model_id}: {str(e)[:100]}")
        
        print(f"\n{simulator.upper()} completed: {completed - failed} successes, {failed} failures")
        return completed - failed, failed
    
    def run_all_simulations_parallel(self):
        """Run simulations using boss/worker pattern - one model type at a time"""
        print(f"Starting {self.num_simulations} simulations with boss/worker pattern")
        
        # Generate parameters first
        if not self.param_maps:
            self.generate_all_param_maps()
        
        model_ids = list(range(self.num_simulations))
        
        # Process each simulator type sequentially
        # DigiSim - one at a time
        print("RUNNING DIGISIM")
        
        # For DigiSim, we need to run one at a time since it's serial
        for idx, model_id in enumerate(model_ids):
            print(f"DigiSim: Processing model {model_id} ({idx+1}/{len(model_ids)})")
            
            param_map = self.param_maps[model_id]
            param_map = compute_preequilibrium(param_map, self.G_with_intermediates)
            
            # Get redox type
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))
            
            if not e_keys:
                print(f"Warning: No E<n> keys for model {model_id}, skipping")
                continue
            
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])
            redox_type = param_map[lowest_e_key]["params"][1]
            
            comsol_params = copy.deepcopy(self.comsol_params)
            if redox_type == "reduction":
                comsol_params.update({
                    'startPotential': 1.0,
                    'vertexPotential1': -1.0,
                    'vertexPotential2': 1.0,
                    'endPotential': 1.0
                })
            elif redox_type == "oxidation":
                comsol_params.update({
                    'startPotential': -1.0,
                    'vertexPotential1': 1.0,
                    'vertexPotential2': -1.0,
                    'endPotential': -1.0
                })
            
            # Run DigiSim (serial)
            try:
                result = run_digisim_simulation(
                    self.G_with_intermediates, param_map, comsol_params, self.scan_rates
                )
                self._save_simulator_results(model_id, 'digisim', result)
                print(f"DigiSim model {model_id} completed")
            except Exception as e:
                self._handle_simulator_error(model_id, 'digisim', str(e))
                print(f"DigiSim model {model_id} failed: {str(e)[:100]}")
        
        # Run ECSIM (parallel)
        print(f"RUNNING ECSIM")
        self._run_model_type('ecsim', self.parallel_executor, model_ids, run_ecsim_simulation)
        
        # Run Electrokitty (parallel)
        print(f"RUNNING ELECTROKITTY")
        self._run_model_type('electrokitty', self.digisim_executor, model_ids, run_electrokitty_simulation)
        
        # Run COMSOL (parallel)
        print(f"RUNNING COMSOL")
        self._run_model_type('comsol', self.parallel_executor, model_ids, run_comsol_simulation)
        
        print("ALL SIMULATIONS COMPLETED!")
    
    def shutdown(self):
        """Clean shutdown of all executors"""
        print("Shutting down executors...")
        self.digisim_executor.shutdown(wait=True)
        self.parallel_executor.shutdown(wait=True)
        print("All executors shut down")

# Utility functions
def generate_randomized_param_map(base_param_map: Dict, random_state: int = 60) -> Dict:
    """Randomize E and C step parameters while preserving structure."""
    print(f"Generating random param map with seed: {random_state}")
    random.seed(random_state)
    np.random.seed(random_state)
    
    new_param_map = deepcopy(base_param_map)
    
    for key, value in base_param_map.items():
        if value['type'] == 'E':
            n = random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2]) # 3 electron steps have been removed
            redox = random.choice(["reduction", "oxidation"]) 
            # Prevent oxidation or reduction the opposite direction
            if int(key[1]) > 0:
                print("I entered the if condition!")
                try:
                    redox = new_param_map["E" + str(int(key[1])-1)]['params'][1]
                except Exception as e:
                    print(e)
                    pass
            E0 = random.uniform(-1, 1)
            k0 = 10**random.uniform(-10, -1)
            alpha = np.clip(np.random.normal(0.5, 0.05),  0.05, 0.95)
            new_param_map[key]['params'] = (n, redox, E0, k0, alpha)
            
        elif value['type'] == 'C':
            while True:
                kf = 10**np.random.uniform(-6, 6)
                kb = 10**np.random.uniform(-6, 6)
                if (np.log10(kf)**2 + np.log10(kb)**2) < 36:
                    break
            new_param_map[key]['params'] = (kf, kb)

    print(f"Generated param map: {new_param_map}")
    
    return new_param_map

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj
