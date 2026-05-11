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
        # Leave 1 core for system, distribute remaining among simulators; they run one after the other, so the remainder can be given to each
        self.parallel_workers = total_cores - 1
        
        # Create executors
        self.parallel_executor = ProcessPoolExecutor(max_workers=self.parallel_workers)
        
        # Simulation parameters
        # constants
        R = 8.314462618  # kg*m^2*s^-2*mol^-1*K^-1
        T = 293.15  # K
        F = 96485.33212  # A*s*mol^-1
        scan_rate = 1
        kc = 1

        logl_min, logl_max = -3, 4
        logg_min, logg_max = -3, 3
        n = 25

        # build centered log grids
        log_l_vals, ret_step = np.linspace(logl_min, logl_max, n, endpoint=False, retstep=True)
        log_l_vals += ret_step / 2

        log_g_vals, ret_step = np.linspace(logg_min, logg_max, n, endpoint=False, retstep=True)
        log_g_vals += ret_step / 2

        LOGL, LOGG = np.meshgrid(log_l_vals, log_g_vals)

        # convert from log space
        lam = 10**LOGL   # l
        g = 10**LOGG     # g

        # compute concentrations
        ccat_grid = lam * scan_rate * F / (R * T * kc)
        csub_grid = g * ccat_grid

        # flatten for storage
        self.ccat_values = ccat_grid.flatten()
        self.csub_values = csub_grid.flatten()
        self.initial_concentration = 1
        self.electrode_radius = 1.0
        self.num_cycles = 1
        self.scan_rates = [scan_rate]
        
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
            'startPotential': 0,
            'numCycles': self.num_cycles,
            'vertexPotential1': 1,
            'vertexPotential2': 0,
            'endPotential': 0,
            'electrodeRadius': self.electrode_radius,
            'startScanRate': 1,
            'endScanRate': 1,
            'scanRateCount': 1
        }
        
        # Storage
        self.param_maps = {}
        self.results = {}
        
        # Create directories
        for model_id in range(self.num_simulations):
            folder_name = f"model_{model_id:04d}"
            os.makedirs(f"simulated_data/{self.mechanism}/{folder_name}", exist_ok=True)
            self.results[model_id] = {
                'comsol': None,
                'folder': folder_name,
                'params_saved': False
            }
    
    def generate_all_param_maps(self):
        print("Generating parameter maps from deterministic grid")

        pairs = list(zip(self.ccat_values, self.csub_values))

        for model_id, (ccat, csub) in enumerate(pairs):

            param_map = generate_randomized_param_map(
                self.base_param_map,
                ccat,
                csub
            )

            self.param_maps[model_id] = param_map

            self._save_parameters(model_id, param_map)
            self.results[model_id]['params_saved'] = True

            print(f"Generated parameters for model {model_id}:  ccat={ccat:.3e} csub={csub:.3e}")
    
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

        # Run COMSOL (parallel)
        print(f"RUNNING COMSOL")
        self._run_model_type('comsol', self.parallel_executor, model_ids, run_comsol_simulation)
        
        print("ALL SIMULATIONS COMPLETED!")
    
    def shutdown(self):
        """Clean shutdown of all executors"""
        print("Shutting down executors...")
        self.digisim_executor.shutdown(wait=True)
        self.parallel_executor.shutdown(wait=True)
        print("All executors shut down")# Utility functions
def generate_randomized_param_map(base_param_map: Dict, kf: float, kb: float) -> Dict:
    """Create param map but use deterministic kf/kb."""
    
    new_param_map = deepcopy(base_param_map)

    for key, value in base_param_map.items():

        if value['type'] == 'E':

            n = 1
            redox = "oxidation"

            if int(key[1]) > 0:
                try:
                    redox = new_param_map["E" + str(int(key[1])-1)]['params'][1]
                except:
                    pass

            E0 = 0.5
            k0 = 0.1
            alpha = 0.5

            new_param_map[key]['params'] = (n, redox, E0, k0, alpha)

        elif value['type'] == 'C':

            # Use deterministic values
            new_param_map[key]['params'] = (kf, kb)

    return new_param_map

def rerun_failed_comsol(base_dir=".", runner=None):
    """
    Scan all model directories for COMSOL failures (CSV files with all-zero data,
    insufficient data, or missing CSV files for expected scan rates),
    rerun the COMSOL simulations using the runner's general simulation functions.
    
    Parameters
    ----------
    base_dir : str
        Path to the folder containing all model directories.
    runner : GeneralSimulationRunner
        An instance of GeneralSimulationRunner to use for rerunning COMSOL.
    """
    if runner is None:
        raise ValueError("Please pass a GeneralSimulationRunner instance as 'runner'.")
    
    if base_dir == '.':
        print("No base directory found. Using the runner's mechanism")
        base_dir = runner.mechanism

    # Find all model directories
    model_dirs = sorted([d for d in os.listdir(base_dir) 
                        if d.startswith("model_") and os.path.isdir(os.path.join(base_dir, d))])

    failed_models = []
    print(f"Scanning {len(model_dirs)} model directories for failed COMSOL simulations...")
    
    # Get expected scan rates from runner
    expected_scan_rates = runner.scan_rates
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        try:
            model_num = int(model_dir.split("_")[1])
        except (ValueError, IndexError):
            print(f"Skipping {model_dir}: invalid model number format")
            continue
            
        # Load parameter map for this model
        param_file = os.path.join(model_path, f"model_{model_num:04d}_params.json")
        if not os.path.exists(param_file):
            print(f"Missing parameter map for {model_dir}, skipping.")
            continue
        
        with open(param_file, 'r') as f:
            param_map = json.load(f)
        
        # Store param_map in runner if not already there
        if model_num not in runner.param_maps:
            runner.param_maps[model_num] = param_map
        
        # Identify all COMSOL CSVs that exist
        existing_comsol_files = [f for f in os.listdir(model_path) 
                               if "comsol_scan_" in f and f.endswith(".csv")]
        
        # Parse scan rates from existing files
        existing_scan_rates = set()
        scan_rate_pattern = re.compile(r"scan_rate_([-\dpeE]+)_V_s\.csv")
        
        for csv_file in existing_comsol_files:
            match = scan_rate_pattern.search(csv_file)
            if match:
                # Convert from filename format (e.g., "0p01" to "0.01")
                scan_str = match.group(1).replace("p", ".")
                try:
                    scan_rate = float(scan_str)
                    existing_scan_rates.add(scan_rate)
                except ValueError:
                    continue
        
        # Check for missing scan rates
        missing_scan_rates = []
        for expected_rate in expected_scan_rates:
            if expected_rate not in existing_scan_rates:
                missing_scan_rates.append(expected_rate)

        # Determine if we need to rerun this model
        needs_rerun = False
        
        # Check existing files for failures
        existing_failures = []
        for csv_file in existing_comsol_files:
            csv_path = os.path.join(model_path, csv_file)
            try:
                df = pd.read_csv(csv_path)
                
                # Check for failure indicators:
                failure_reason = None
                if "i" not in df.columns:
                    needs_rerun = True
                    failure_reason = "Missing 'i' column"
                elif len(df) < 100:
                    needs_rerun = True
                    failure_reason = f"Insufficient data points ({len(df)} < 100)"
                
                if failure_reason:
                    # Try to extract scan rate for logging
                    match = scan_rate_pattern.match(csv_file)
                    if match:
                        scan_str = match.group(1).replace("p", ".")
                        try:
                            scan_rate = float(scan_str)
                            existing_failures.append((scan_rate, failure_reason))
                        except ValueError:
                            existing_failures.append((csv_file, failure_reason))
                    
            except Exception as e:
                # File might be corrupted or empty
                existing_failures.append((csv_file, f"Read error: {str(e)}"))
                needs_rerun = True
        
        if missing_scan_rates:
            print(f"Model {model_dir}: Missing {len(missing_scan_rates)} scan rate files:")
            for rate in missing_scan_rates:
                print(f"  - {rate:.2e}")
            needs_rerun = True
        
        if existing_failures:
            print(f"Model {model_dir}: Found {len(existing_failures)} failed CSV files:")
            for scan_rate, reason in existing_failures:
                if isinstance(scan_rate, (int, float)):
                    print(f"  - Scan rate {scan_rate:.2e}: {reason}")
                else:
                    print(f"  - File {scan_rate}: {reason}")
            needs_rerun = True
        
        # If we need to rerun, add to failed models list
        if needs_rerun:
            failed_models.append(model_num)
            
            # Print the reason
            if existing_failures:
                print(f"  Marking model {model_num} for rerun due to CSV failures")
            elif missing_scan_rates:
                print(f"  Marking model {model_num} for rerun due to missing scan rates")
    
    # Rerun failed models using the runner's _run_model_type method
    if failed_models:
        print(f"\nRerunning COMSOL for {len(failed_models)} failed models: {failed_models}")
        
        # Use the runner's parallel executor to rerun failed models
        successes, failures = runner._run_model_type(
            'comsol', 
            runner.parallel_executor, 
            failed_models, 
            run_comsol_simulation
        )
        
        print(f"\nCOMSOL rerun completed:")
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        
        return successes, failures
    else:
        print("\nNo failed COMSOL simulations found!")
        return 0, 0

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