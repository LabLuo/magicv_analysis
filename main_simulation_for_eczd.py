import threading
import concurrent.futures
from queue import Queue, Empty
import time
from typing import List, Callable, Any, Dict
import os
import json
import pandas as pd
import numpy as np
from copy import deepcopy
import random
import multiprocessing as mp
from threading import Semaphore, Lock
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import re
import gc

# Assuming these imports are available in your environment
from simulation_utils import *

class MultithreadingManager:
    def __init__(self, total_cores: int = 8):
        self.total_cores = total_cores
        self.comsol_timeout = 300
        
        # Dynamic core allocation - prioritizing parallel execution
        # Reserve 1 core for system/overhead, distribute rest
        available_cores = total_cores - 1
        
        # Allocate cores with focus on parallelism
        self.core_allocation = {
            'comsol': available_cores  # Remaining cores for COMSOL
        }
        
        print(f"Core allocation: {self.core_allocation}")
        
        # Thread/Process pools for each simulator
        # COMSOL: Use all allocated cores for each task
        self.comsol_pool = ProcessPoolExecutor(max_workers=self.core_allocation['comsol'])
        
        # Queues for each simulator
        self.comsol_queue = Queue()
        
        # Track active tasks
        self.active_tasks = {
            'comsol': 0
        }
        
        # Locks for thread-safe operations
        self.active_tasks_lock = Lock()
        
        # Flags to control queue processing
        self.stop_queues = False
        
        # Start queue processors
        self.queue_processors = []
        self.start_all_queue_processors()
    
    def _update_active_tasks(self, simulator: str, delta: int):
        """Thread-safe update of active task count"""
        with self.active_tasks_lock:
            self.active_tasks[simulator] += delta
            if self.active_tasks[simulator] < 0:
                self.active_tasks[simulator] = 0
    
    def start_all_queue_processors(self):
        """Start independent queue processors for each simulator"""
        self.queue_processors = [
            threading.Thread(target=self._process_comsol_queue, daemon=True)
        ]
        
        for processor in self.queue_processors:
            processor.start()
    
    def submit_task(self, simulator: str, task_data: Dict):
        """Submit task to appropriate queue"""
        if simulator == 'digisim':
            self.digisim_queue.put(task_data)
        elif simulator == 'ecsim':
            self.ecsim_queue.put(task_data)
        elif simulator == 'comsol':
            self.comsol_queue.put(task_data)
        elif simulator == 'electrokitty':
            self.electrokitty_queue.put(task_data)
        else:
            raise ValueError(f"Unknown simulator: {simulator}")

    def _process_comsol_queue(self):
        """Process comsol queue with timeout and limit to 8 concurrent processes"""
        comsol_semaphore = threading.Semaphore(8)

        def process_single_task(task_data):
            comsol_semaphore.acquire()
            try:
                print(
                    f"COMSOL: Starting model {task_data['model_id']} "
                    f"(Active: {8 - comsol_semaphore._value})"
                )

                self._update_active_tasks('comsol', 1)

                future = self.comsol_pool.submit(
                    task_data['func'],
                    *task_data['args'],
                    **task_data['kwargs']
                )

                def handle_future_with_timeout(f, td=task_data):
                    try:
                        # Wait for completion or timeout
                        f.result(timeout=self.comsol_timeout)
                        print(f"COMSOL task {td['model_id']} proved successful.")
                    except concurrent.futures.TimeoutError:
                        print(
                            f"COMSOL task {td['model_id']} "
                            f"timed out after {self.comsol_timeout}s"
                        )
                        # Cancel if possible
                        f.cancel()
                    finally:
                        # Centralized completion handling
                        self._handle_task_completion(f, td, 'comsol')
                        comsol_semaphore.release()

                threading.Thread(
                    target=handle_future_with_timeout,
                    args=(future,),
                    daemon=True
                ).start()

            except Exception as e:
                comsol_semaphore.release()
                print(f"Error submitting COMSOL task {task_data['model_id']}: {e}")
                self._handle_task_completion_error(task_data, 'comsol', str(e))
                self.comsol_queue.task_done()

        while not self.stop_queues:
            try:
                task_data = self.comsol_queue.get(timeout=1)
                threading.Thread(
                    target=process_single_task,
                    args=(task_data,),
                    daemon=True
                ).start()
            except Empty:
                continue
            except Exception as e:
                print(f"Error in comsol queue processor: {e}")
    
    def _handle_task_completion(self, future, task_data, simulator):
        """Handle task completion (success or error)"""
        model_id = task_data['model_id']

        print(f"A task ({model_id}) has completed in Lego City.")
        
        try:
            # Decrement active task count
            self._update_active_tasks(simulator, -1)
            
            # Get result (this will raise if there was an exception)
            result = future.result(timeout=0.1)
            
            # Call success callback
            print("I'm now doing the callback")
            task_data['callback'](model_id, simulator, result)
            
        except concurrent.futures.TimeoutError:
            # Task might still be running
            print(f"Warning: {simulator} task for model {model_id} timed out during cleanup")
        except Exception as e:
            # Call error callback
            task_data['error_callback'](model_id, simulator, str(e))
        finally:
            # Mark queue task as done
            if simulator == 'digisim':
                self.digisim_queue.task_done()
            elif simulator == 'ecsim':
                self.ecsim_queue.task_done()
            elif simulator == 'comsol':
                self.comsol_queue.task_done()
            elif simulator == 'electrokitty':
                self.electrokitty_queue.task_done()
    
    def shutdown(self):
        """Clean shutdown of all thread pools"""
        self.stop_queues = True
        
        # Wait for active tasks to complete
        print("Waiting for active tasks to complete...")
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            active_total = sum(self.active_tasks.values())
            if active_total == 0:
                break
            print(f"Waiting for {active_total} active tasks to complete...")
            time.sleep(5)
        
        # Force shutdown if tasks are stuck
        print("Shutting down pools...")
        
        # Shutdown pools
        self.digisim_pool.shutdown(wait=True, cancel_futures=True)
        self.ecsim_pool.shutdown(wait=True, cancel_futures=True)
        self.comsol_pool.shutdown(wait=True, cancel_futures=True)
        self.electrokitty_pool.shutdown(wait=True, cancel_futures=True)

class GeneralSimulationRunner:
    def __init__(self, mechanism: str = "ECE", num_simulations: int = 250, 
                 random_state: int = 60, total_cores: int = 8):
        # constants
        R = 8.314
        T = 298.15
        F = 96485
        scan_rate = 1

        logd_min, logd_max = -2-1, 4.5-1
        logK_min, logK_max = -1, 3
        n = 25

        log_d_vals, ret_step = np.linspace(logd_min, logd_max, n, endpoint=False, retstep=True)
        log_d_vals += ret_step / 2

        log_K_vals, ret_step = np.linspace(logK_min, logK_max, n, endpoint=False, retstep=True)
        log_K_vals += ret_step / 2

        LOGD, LOGK = np.meshgrid(log_d_vals, log_K_vals)

        lam = 10**LOGD
        K = 10**LOGK

        S = lam * scan_rate * F / (R * T)

        kb_grid = S / (K + 1)
        kf_grid = K * kb_grid

        self.kf_values = kf_grid.flatten()
        self.kb_values = kb_grid.flatten()

        # Old stuff
        self.mechanism = mechanism
        self.num_simulations = num_simulations
        self.random_state = random_state
        self.total_cores = total_cores
        
        # Set process start method for ProcessPoolExecutor
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Initialize thread manager
        self.thread_manager = MultithreadingManager(total_cores)
        
        # Simulation parameters
        self.scan_rates = [1]
        self.initial_concentration = 1
        self.electrode_radius = 1.0
        self.num_cycles = 1
        
        # Get the graph
        text = self.mechanism
        parser = SynthesisParser(text)
        G = parser.parse()
        parser.draw()
        
        # Insert intermediates, starting materials, and products
        self.G_with_intermediates = insert_intermediates(G)

        # Create the parameter map that matches the suggested mechanism
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
        
        # Storage for parameter maps
        self.param_maps = {}
        
        # Storage for results (organized by model_id and simulator)
        self.results = {}
        
        # Create model directories
        for model_id in range(self.num_simulations):
            folder_name = f"model_{model_id:04d}"
            os.makedirs(f"new/{self.mechanism}/{folder_name}", exist_ok=True)
            self.results[model_id] = {
                'digisim': None,
                'ecsim': None,
                'electrokitty': None,
                'comsol': None,
                'folder': folder_name,
                'params_saved': False
            }
    
    def generate_all_param_maps(self):

        print("Generating parameter maps from deterministic kf/kb grid")

        pairs = list(zip(self.kf_values, self.kb_values))

        for model_id, (kf, kb) in enumerate(pairs):

            param_map = generate_randomized_param_map(
                self.base_param_map,
                kf,
                kb
            )

            self.param_maps[model_id] = param_map

            self._save_parameters(model_id, param_map)
            self.results[model_id]['params_saved'] = True

            print(f"Generated parameters for model {model_id}  kf={kf:.3e} kb={kb:.3e}")
    
    def _save_parameters(self, model_id: int, param_map: Dict):
        """Save parameters for a single model"""
        folder_name = self.results[model_id]['folder']
        param_filename = f"new/{self.mechanism}/{folder_name}/model_{model_id:04d}_params.json"
        
        with open(param_filename, 'w') as f:
            json_ready = convert_numpy_types(param_map)
            json.dump(json_ready, f, indent=2)
    
    def _save_simulator_results(self, model_id: int, simulator: str, results: Dict):
        """Save results for a single simulator as soon as they're available"""
        print("I successfully entered the save.")
        if results is None:
            print("But I have no results")
            return
            
        folder_name = self.results[model_id]['folder']
        
        # Process results for each scan rate
        for scan_key, data in results.items():
            if 'potential' in data.keys() and 'current' in data.keys():
                # Create DataFrame
                df_data = {
                    'E': data['potential'],
                    'i': data['current']
                }
                
                # Add time if available (for COMSOL)
                if 'time' in data:
                    df_data['t'] = data['time']
                
                df = pd.DataFrame(df_data)
                
                # Clean up scan_key for filename
                filename_scan_key = str(scan_key).replace('.', 'p').replace('+', '')
                filename = f"new/{self.mechanism}/{folder_name}/model_{model_id:04d}_{simulator}_scan_{filename_scan_key}.csv"
                print(f"I am saving to {filename}")
                df.to_csv(filename, index=False)
        
        print(f"Saved {simulator} results for model {model_id}")
    
    def _handle_simulator_error(self, model_id: int, simulator: str, error: str):
        """Handle errors from simulators"""
        print(f"Error in {simulator} for model {model_id}: {error}")
        # Log error to file
        folder_name = self.results[model_id]['folder']
        error_filename = f"new/{self.mechanism}/{folder_name}/model_{model_id:04d}_{simulator}_error.txt"
        with open(error_filename, 'w') as f:
            f.write(f"Error in {simulator}: {error}\n")
        gc.collect()
    
    def run_all_simulations_parallel(self):
        """Run all simulations with independent parallel queues"""
        print(f"Starting {self.num_simulations} simulations with parallel queues")
        
        # Generate parameters first if not already done
        if not self.param_maps:
            self.generate_all_param_maps()
        
        # Submit all tasks to queues
        for model_id in range(self.num_simulations):
            param_map = self.param_maps[model_id]

            param_map = compute_preequilibrium(param_map, self.G_with_intermediates)

            # Find all E<n> keys
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))

            if not e_keys:
                raise ValueError(f"No E<n> keys found for model {model_id}")

            # Select the E with the lowest index
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])

            redox_type = param_map[lowest_e_key]["params"][1]

            comsol_params = copy.deepcopy(self.comsol_params)

            # Submit comsol task (gets all allocated cores)
            self.thread_manager.submit_task('comsol', {
                'model_id': model_id,
                'func': run_comsol_simulation,
                'args': (self.G_with_intermediates, param_map, None, comsol_params, self.scan_rates),
                'kwargs': {},
                'callback': self._save_simulator_results,
                'error_callback': self._handle_simulator_error
            })
        
        print("All tasks submitted to queues. Processing in parallel...")
        print(f"COMSOL gets {self.thread_manager.core_allocation['comsol']} cores per task")
        
        # Wait for all queues to process
        self._wait_for_queues_to_empty()
        
        print("All simulations completed!")
    
    def _wait_for_queues_to_empty(self):
        """Wait for all queues to empty"""
        # Simple polling approach
        repeat_count = 0
        
        while True:
            time.sleep(10)
            active_tasks = self.thread_manager.active_tasks.copy()

            # Get queue sizes
            q_sizes = {
                'comsol': self.thread_manager.comsol_queue.qsize(),
            }
            
            print(f"COMSOL={q_sizes['comsol']}(A:{active_tasks['comsol']})")
            
            print(f"The number of active tasks is {sum(active_tasks.values())} while the repeat count is {repeat_count}")
            
            # Check if all queues are empty and no active tasks
            if (sum(active_tasks.values()) == 0 or repeat_count > 10):
                print("All queues empty and no active tasks")
                break
    
    def shutdown(self):
        """Clean shutdown"""
        self.thread_manager.shutdown()

def is_outside_region(kf, kb):
    """
    Return True if point (kf, kb) is outside the region defined by:
    - For kb <= -9:     kf >= -kb - 17
    - For -9 < kb <= 7: kf >= -8
    - For kb > 7:       kf >= kb - 15
    """
    kf_log = np.log10(kf) if kf > 0 else -20
    kb_log = np.log10(kb) if kb > 0 else -20
    
    if kb_log <= -9:
        return kf_log >= -kb_log - 17
    elif -9 < kb_log <= 5:
        return kf_log >= 8
    else:  # kb_log > 5
        return kf_log >= -kb_log + 14

# Utility functions
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
    rerun the COMSOL simulations using the runner's thread manager.
    
    Parameters
    ----------
    base_dir : str
        Path to the folder containing all model directories.
    runner : GeneralSimulationRunner
        An instance of GeneralSimulationRunner to use for rerunning COMSOL.
    """
    if runner is None:
        raise ValueError("Please pass an GeneralSimulationRunner instance as 'runner'.")
    
    if base_dir == '.':
        print("No base directory found. Using the runner's mechanism")
        base_dir = runner.mechanism

    # Find all model directories
    model_dirs = sorted([d for d in os.listdir(base_dir) 
                        if d.startswith("model_") and os.path.isdir(os.path.join(base_dir, d))])

    rerun_count = 0
    failed_count = 0
    submissions = []

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
        
        # If we need to rerun, prepare and submit the task
        if needs_rerun:
            rerun_count += 1

            try:
                if failure_reason:
                    print(f"Rerunning due to failure: {failure_reason}")
            except:
                pass
            
            # Find all E<n> keys
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))

            if not e_keys:
                print(f"  Warning: No E<n> keys found in param map for {model_dir}, skipping rerun")
                continue
            
            # Select the E with the lowest index
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])
            redox_type = param_map[lowest_e_key]["params"][1]

            comsol_params = copy.deepcopy(runner.comsol_params)

            # Update comsol_params based on redox type
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
            
            # Submit COMSOL task through the runner's thread manager
            print(f"  Submitting rerun for model {model_num}...")
            
            task_data = {
                'model_id': model_num,
                'func': run_comsol_simulation,
                'args': (runner.G_with_intermediates, param_map, None, comsol_params, runner.scan_rates),
                'kwargs': {},
                'callback': runner._save_simulator_results,
                'error_callback': runner._handle_simulator_error
            }
            
            # Store submission info
            submissions.append(task_data)
            
            # Submit the task
            runner.thread_manager.submit_task('comsol', task_data)

    if rerun_count > 0:
        print(f"\nSubmitted {rerun_count} models for COMSOL rerun")
        print("Waiting for reruns to complete...")
        
        # Wait for COMSOL queue to empty
        start_time = time.time()
        max_wait_time = 1800  # 30 minutes maximum wait
        
        while time.time() - start_time < max_wait_time:
            time.sleep(10)
            
            # Get queue and task status
            q_size = runner.thread_manager.comsol_queue.qsize()
            active_tasks = runner.thread_manager.active_tasks['comsol']
            
            print(f"  COMSOL queue: {q_size} pending, {active_tasks} active")
            
            if q_size == 0 and active_tasks == 0:
                print("  All reruns completed!")
                break
        else:
            print(f"  Warning: Timeout waiting for reruns to complete")
    
    # Final check for failures after rerun
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        try:
            model_num = int(model_dir.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # Check if there's an error file from the rerun
        error_file = os.path.join(model_path, f"model_{model_num:04d}_comsol_error.txt")
        
        if os.path.exists(error_file):
            failed_count += 1
            print(f"⚠️  Model {model_num} still has errors after rerun")
        
        # Also check if all expected scan rates now exist
        existing_comsol_files = [f for f in os.listdir(model_path) 
                               if "comsol_scan_" in f and f.endswith(".csv")]
        
    print(f"\nSummary:")
    print(f"  Total models submitted for rerun: {rerun_count}")
    print(f"  Models still failing after rerun: {failed_count}")
    print(f"  Expected scan rates: {len(expected_scan_rates)}")
    
    return rerun_count, failed_count

def rerun_failed_digisim(base_dir=".", runner=None):
    if runner is None:
        raise ValueError("Please pass an GeneralSimulationRunner instance as 'runner'.")
    
    if base_dir == '.':
        print("No base directory found. Using the runner's mechanism")
        base_dir = runner.mechanism

    # Find all model directories
    model_dirs = sorted([d for d in os.listdir(base_dir) 
                        if d.startswith("model_") and os.path.isdir(os.path.join(base_dir, d))])

    rerun_count = 0
    failed_count = 0
    submissions = []

    print(f"Scanning {len(model_dirs)} model directories for failed digisim simulations...")
    
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
        
        # Identify all digisim CSVs that exist
        existing_digisim_files = [f for f in os.listdir(model_path) 
                               if "digisim_scan_" in f and f.endswith(".csv")]
        
        # Parse scan rates from existing files
        existing_scan_rates = set()
        scan_rate_pattern = re.compile(r"scan_rate_([-\dpeE]+)_V_s\.csv")
        
        for csv_file in existing_digisim_files:
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
        for csv_file in existing_digisim_files:
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
                failure_reason = "Read error"
        
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
        
        # If we need to rerun, prepare and submit the task
        if True: #if needs_rerun:
            rerun_count += 1

            try:
                if failure_reason:
                    print(f"Rerunning due to failure: {failure_reason}")
            except:
                pass
            
            # Find all E<n> keys
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))

            if not e_keys:
                print(f"  Warning: No E<n> keys found in param map for {model_dir}, skipping rerun")
                continue
            
            # Select the E with the lowest index
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])
            redox_type = param_map[lowest_e_key]["params"][1]

            comsol_params = copy.deepcopy(runner.comsol_params)

            # Update comsol_params based on redox type
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
            
            # Submit Digisim task through the runner's thread manager
            print(f"  Submitting rerun for model {model_num}...")
            
            task_data = {
                'model_id': model_num,
                'func': run_digisim_simulation,
                'args': (runner.G_with_intermediates, param_map, None, comsol_params, runner.scan_rates),
                'kwargs': {},
                'callback': runner._save_simulator_results,
                'error_callback': runner._handle_simulator_error
            }
            
            # Store submission info
            submissions.append(task_data)
            
            # Submit the task
            runner.thread_manager.submit_task('digisim', task_data)

    if rerun_count > 0:
        print(f"\nSubmitted {rerun_count} models for Digisim rerun")
        print("Waiting for reruns to complete...")
        
        # Wait for Digisim queue to empty
        start_time = time.time()
        max_wait_time = 1800  # 30 minutes maximum wait
        
        while time.time() - start_time < max_wait_time:
            time.sleep(10)
            
            # Get queue and task status
            q_size = runner.thread_manager.digisim_queue.qsize()
            active_tasks = runner.thread_manager.active_tasks['digisim']
            
            print(f"  Digisim queue: {q_size} pending, {active_tasks} active")
            
            if q_size == 0 and active_tasks == 0:
                print("  All reruns completed!")
                break
        else:
            print(f"  Warning: Timeout waiting for reruns to complete")
    
    # Final check for failures after rerun
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        try:
            model_num = int(model_dir.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # Check if there's an error file from the rerun
        error_file = os.path.join(model_path, f"model_{model_num:04d}_digisim_error.txt")
        
        if os.path.exists(error_file):
            failed_count += 1
            print(f"⚠️  Model {model_num} still has errors after rerun")
        
    print(f"\nSummary:")
    print(f"  Total models submitted for rerun: {rerun_count}")
    print(f"  Models still failing after rerun: {failed_count}")
    print(f"  Expected scan rates: {len(expected_scan_rates)}")
    
    return rerun_count, failed_count

def rerun_failed_electrokitty(base_dir=".", runner=None):
    if runner is None:
        raise ValueError("Please pass an GeneralSimulationRunner instance as 'runner'.")
    
    if base_dir == '.':
        print("No base directory found. Using the runner's mechanism")
        base_dir = runner.mechanism

    # Find all model directories
    model_dirs = sorted([d for d in os.listdir(base_dir) 
                        if d.startswith("model_") and os.path.isdir(os.path.join(base_dir, d))])

    rerun_count = 0
    failed_count = 0
    submissions = []

    print(f"Scanning {len(model_dirs)} model directories for failed Electrokitty simulations...")
    
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
        
        # Identify all Electrokitty CSVs that exist
        existing_electrokitty_files = [f for f in os.listdir(model_path) 
                               if "electrokitty_scan_" in f and f.endswith(".csv")]
        
        # Parse scan rates from existing files
        existing_scan_rates = set()
        scan_rate_pattern = re.compile(r"scan_rate_([-\dpeE]+)_V_s\.csv")
        
        for csv_file in existing_electrokitty_files:
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
        for csv_file in existing_electrokitty_files:
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
        
        # If we need to rerun, prepare and submit the task
        if needs_rerun:
            rerun_count += 1

            try:
                if failure_reason:
                    print(f"Rerunning due to failure: {failure_reason}")
            except:
                pass
            
            # Find all E<n> keys
            e_keys = []
            for key in param_map.keys():
                match = re.fullmatch(r"E(\d+)", key)
                if match:
                    e_keys.append((int(match.group(1)), key))

            if not e_keys:
                print(f"  Warning: No E<n> keys found in param map for {model_dir}, skipping rerun")
                continue
            
            # Select the E with the lowest index
            _, lowest_e_key = min(e_keys, key=lambda x: x[0])
            redox_type = param_map[lowest_e_key]["params"][1]

            comsol_params = copy.deepcopy(runner.comsol_params)

            # Update comsol_params based on redox type
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

            param_map = compute_preequilibrium(param_map, runner.G_with_intermediates)
            
            # Submit Electrokitty task through the runner's thread manager
            print(f"  Submitting rerun for model {model_num}...")
            
            task_data = {
                'model_id': model_num,
                'func': run_electrokitty_simulation,
                'args': (runner.G_with_intermediates, param_map, None, comsol_params, runner.scan_rates),
                'kwargs': {},
                'callback': runner._save_simulator_results,
                'error_callback': runner._handle_simulator_error
            }
            
            # Store submission info
            submissions.append(task_data)
            
            # Submit the task
            runner.thread_manager.submit_task('electrokitty', task_data)

    if rerun_count > 0:
        print(f"\nSubmitted {rerun_count} models for Electrokitty rerun")
        print("Waiting for reruns to complete...")
        
        # Wait for Electrokitty queue to empty
        start_time = time.time()
        max_wait_time = 1800  # 30 minutes maximum wait
        
        while time.time() - start_time < max_wait_time:
            time.sleep(10)
            
            # Get queue and task status
            q_size = runner.thread_manager.electrokitty_queue.qsize()
            active_tasks = runner.thread_manager.active_tasks['electrokitty']
            
            print(f"  Electrokitty queue: {q_size} pending, {active_tasks} active")
            
            if q_size == 0 and active_tasks == 0:
                print("  All reruns completed!")
                break
        else:
            print(f"  Warning: Timeout waiting for reruns to complete")
    
    # Final check for failures after rerun
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        try:
            model_num = int(model_dir.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # Check if there's an error file from the rerun
        error_file = os.path.join(model_path, f"model_{model_num:04d}_electrokitty_error.txt")
        
        if os.path.exists(error_file):
            failed_count += 1
            print(f"⚠️  Model {model_num} still has errors after rerun")
        
    print(f"\nSummary:")
    print(f"  Total models submitted for rerun: {rerun_count}")
    print(f"  Models still failing after rerun: {failed_count}")
    print(f"  Expected scan rates: {len(expected_scan_rates)}")
    
    return rerun_count, failed_count

def rerun_list_comsol(model_dirs, runner, base_dir="."):
    """
    Rerun COMSOL simulations for a specific list of model directories.
    
    Parameters
    ----------
    model_dirs : list
        List of model directory names or IDs to rerun (e.g., ["model_0001", "model_0002"] 
        or [1, 2, 3])
    runner : ECESimulationRunner
        An instance of ECESimulationRunner to use for rerunning COMSOL.
    base_dir : str
        Path to the folder containing all model directories.
    
    Returns
    -------
    tuple
        (success_count, failure_count, error_list)
    """
    if runner is None:
        raise ValueError("Please pass an ECESimulationRunner instance as 'runner'.")
    
    if not model_dirs:
        print("No model directories specified for rerun.")
        return 0, 0, []
    
    print(f"Preparing to rerun COMSOL simulations for {len(model_dirs)} specified models...")
    
    submissions = []
    error_list = []
    model_info = {}
    
    # Process the input list - it could be directory names or numeric IDs
    for model_item in model_dirs:
        if isinstance(model_item, str):
            # It's a directory name like "model_0001"
            model_dir = model_item
            try:
                model_num = int(model_dir.split("_")[1])
                model_path = os.path.join(base_dir, model_dir)
                model_info[model_num] = {
                    'dir': model_dir,
                    'path': model_path,
                    'num': model_num
                }
            except (ValueError, IndexError):
                print(f"Skipping {model_item}: invalid model directory format")
                error_list.append(f"Invalid format: {model_item}")
                continue
                
        elif isinstance(model_item, int):
            # It's a numeric ID like 1, 2, 3
            model_num = model_item
            model_dir = f"model_{model_num:04d}"
            model_path = os.path.join(base_dir, model_dir)
            model_info[model_num] = {
                'dir': model_dir,
                'path': model_path,
                'num': model_num
            }
        else:
            print(f"Skipping {model_item}: invalid type {type(model_item)}")
            error_list.append(f"Invalid type: {model_item}")
            continue
    
    # Submit tasks for each model
    for model_num, info in model_info.items():
        model_path = info['path']
        
        # Check if directory exists
        if not os.path.exists(model_path):
            print(f"Skipping model {model_num}: directory {info['dir']} not found")
            error_list.append(f"Directory not found: {info['dir']}")
            continue
            
        # Load parameter map
        param_file = os.path.join(model_path, f"model_{model_num:04d}_params.json")
        if not os.path.exists(param_file):
            print(f"Missing parameter map for model {model_num}, skipping.")
            error_list.append(f"Missing params file: {info['dir']}")
            continue
        
        try:
            with open(param_file, 'r') as f:
                param_map = json.load(f)
        except Exception as e:
            print(f"Error loading params for model {model_num}: {e}")
            error_list.append(f"Error loading params for {info['dir']}: {str(e)}")
            continue
        
        # Find all E<n> keys
        e_keys = []
        for key in param_map.keys():
            match = re.fullmatch(r"E(\d+)", key)
            if match:
                e_keys.append((int(match.group(1)), key))

        # Select the E with the lowest index
        _, lowest_e_key = min(e_keys, key=lambda x: x[0])

        redox_type = param_map[lowest_e_key]["params"][1]

        comsol_params = copy.deepcopy(runner.comsol_params)

        # Update comsol_params based on redox type
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
        
        # Submit COMSOL task through the runner's thread manager
        print(f"  Submitting rerun for model {model_num} ({info['dir']})...")
        
        task_data = {
            'model_id': model_num,
            'func': run_comsol_simulation,
            'args': (runner.G_with_intermediates, param_map, None, comsol_params, runner.scan_rates),
            'kwargs': {},
            'callback': runner._save_simulator_results,
            'error_callback': runner._handle_simulator_error
        }
        
        # Store submission info
        submissions.append({
            'model_num': model_num,
            'dir': info['dir'],
            'task_data': task_data
        })
        
        # Submit the task
        runner.thread_manager.submit_task('comsol', task_data)
    
    # Wait for tasks to complete if any were submitted
    if submissions:
        print(f"\nSubmitted {len(submissions)} models for COMSOL rerun")
        print("Waiting for reruns to complete...")
        
        # Wait for COMSOL queue to empty
        start_time = time.time()
        max_wait_time = 1800  # 30 minutes maximum wait
        
        while time.time() - start_time < max_wait_time:
            time.sleep(10)
            
            # Get queue and task status
            q_size = runner.thread_manager.comsol_queue.qsize()
            active_tasks = runner.thread_manager.active_tasks['comsol']
            
            print(f"  COMSOL queue: {q_size} pending, {active_tasks} active")
            
            if q_size == 0 and active_tasks == 0:
                print("  All reruns completed!")
                break
        else:
            print(f"  Warning: Timeout waiting for reruns to complete")
    
    # Check results and count successes/failures
    success_count = 0
    failure_count = 0
    
    for sub in submissions:
        model_num = sub['model_num']
        model_dir = sub['dir']
        model_path = sub['task_data']['model_path'] if 'model_path' in sub['task_data'] else os.path.join(base_dir, model_dir)
        
        # Check for error files
        error_file = os.path.join(model_path, f"model_{model_num:04d}_comsol_error.txt")
        if os.path.exists(error_file):
            failure_count += 1
            print(f"❌ Model {model_num} ({model_dir}) failed - see error file")
            error_list.append(f"Failed: {model_dir}")
        else:
            # Check if CSV files were created successfully
            comsol_files = [f for f in os.listdir(model_path) 
                          if f"model_{model_num:04d}_comsol_scan_" in f and f.endswith(".csv")]
            
            if len(comsol_files) == len(runner.scan_rates):
                success_count += 1
                print(f"✅ Model {model_num} ({model_dir}) completed successfully")
            else:
                failure_count += 1
                print(f"⚠️  Model {model_num} ({model_dir}) incomplete - only {len(comsol_files)}/{len(runner.scan_rates)} scan rates")
                error_list.append(f"Incomplete: {model_dir}")
    
    print(f"\nSummary:")
    print(f"  Total models submitted: {len(submissions)}")
    print(f"  Successful reruns: {success_count}")
    print(f"  Failed reruns: {failure_count}")
    
    if error_list:
        print(f"\nErrors encountered:")
        for error in error_list:
            print(f"  - {error}")
    
    return success_count, failure_count, error_list

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
