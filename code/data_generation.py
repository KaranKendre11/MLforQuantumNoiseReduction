import cirq
import random
import numpy as np
from tqdm import tqdm
import scipy.linalg

def generate_random_circuit(num_qubits, depth, seed=None):
    """
    Generate a random quantum circuit with specified number of qubits and depth.
    
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Number of layers of gates
        seed: Random seed for reproducibility
    
    Returns:
        A random Cirq circuit
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    
    # Gates to sample from
    single_qubit_gates = [
        cirq.X, cirq.Y, cirq.Z,
        cirq.H, cirq.T, cirq.S,
        lambda q: cirq.rx(np.random.uniform(0, 2*np.pi))(q),
        lambda q: cirq.ry(np.random.uniform(0, 2*np.pi))(q),
        lambda q: cirq.rz(np.random.uniform(0, 2*np.pi))(q)
    ]
    
    two_qubit_gates = [
        cirq.CZ, cirq.CNOT,
        lambda q1, q2: cirq.SWAP(q1, q2)
    ]
    
    for d in range(depth):
        # Add single-qubit gates
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))
        
        # Add two-qubit gates (to create entanglement)
        qubit_pairs = list(zip(qubits[:-1], qubits[1:]))  # Adjacent qubits
        for q1, q2 in random.sample(qubit_pairs, k=min(len(qubit_pairs), num_qubits//2)):
            gate = random.choice(two_qubit_gates)
            circuit.append(gate(q1, q2))
    
    return circuit

def add_noise(circuit, noise_type, noise_level):
    """
    Add specified noise model to a quantum circuit.
    
    Args:
        circuit: The original quantum circuit
        noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'phase_damping', 'bitflip', 'mixed')
        noise_level: The strength of the noise (0 to 1)
    
    Returns:
        A noisy quantum circuit
    """
    qubits = sorted(circuit.all_qubits())
    noisy_circuit = cirq.Circuit()
    
    if noise_type == 'depolarizing':
        # Depolarizing noise: replaces the qubit state with a completely mixed state with probability p
        noise_model = cirq.depolarize(p=noise_level)
        
    elif noise_type == 'amplitude_damping':
        # Amplitude damping: models energy dissipation (e.g., spontaneous emission)
        noise_model = cirq.amplitude_damp(gamma=noise_level)
        
    elif noise_type == 'phase_damping':
        # Phase damping: models loss of quantum information without energy dissipation
        noise_model = cirq.phase_damp(gamma=noise_level)
        
    elif noise_type == 'bitflip':
        # Bit flip: flips the state of a qubit with probability p
        noise_model = cirq.bit_flip(p=noise_level)
        
    elif noise_type == 'mixed':
        # Combination of different noise types
        noise_models = [
            cirq.depolarize(p=noise_level/3),
            cirq.amplitude_damp(gamma=noise_level/3),
            cirq.phase_damp(gamma=noise_level/3)
        ]
    
    # Create the noisy circuit by inserting noise after each operation
    for moment in circuit:
        noisy_circuit.append(moment)
        for q in qubits:
            if noise_type == 'mixed':
                for noise in noise_models:
                    noisy_circuit.append(noise.on(q))
            else:
                noisy_circuit.append(noise_model.on(q))
    
    return noisy_circuit

def circuit_to_density_matrix(circuit, num_qubits):
    """
    Convert a circuit to its density matrix representation.
    
    Args:
        circuit: The quantum circuit
        num_qubits: Number of qubits in the circuit
    
    Returns:
        The density matrix as a complex numpy array
    """
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit)
    density_matrix = result.final_density_matrix
    
    # Reshape for CNN input (2 channels for real and imaginary parts)
    dim = 2**num_qubits
    density_matrix_reshaped = density_matrix.reshape(dim, dim)
    
    return density_matrix_reshaped

def prepare_dataset(num_circuits, num_qubits, noise_types, noise_levels, seed=None):
    """
    Generate a dataset of clean and noisy quantum circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        num_qubits: Number of qubits per circuit
        noise_types: List of noise types to use
        noise_levels: List of noise levels to use
        seed: Random seed for reproducibility
    
    Returns:
        X_clean: Clean circuit density matrices
        X_noisy: Noisy circuit density matrices
        y: Labels (noise types and levels)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    X_clean = []
    X_noisy = []
    y = []
    
    for i in tqdm(range(num_circuits)):
        # Generate a random seed for each circuit
        circuit_seed = random.randint(0, 10000)
        circuit_depth = random.randint(6, 9)
        
        # Generate a clean circuit
        clean_circuit = generate_random_circuit(num_qubits, circuit_depth, seed=circuit_seed)
        clean_dm = circuit_to_density_matrix(clean_circuit, num_qubits)
        
        
        # Add different types of noise
        noise_type = random.choice(noise_types)
        noise_level = random.choice(noise_levels)
        
        noisy_circuit = add_noise(clean_circuit, noise_type, noise_level)
        noisy_dm = circuit_to_density_matrix(noisy_circuit, num_qubits)
        
        # Store the data
        X_clean.append(clean_dm)
        X_noisy.append(noisy_dm)
        y.append((noise_type, noise_level))
    
    return np.array(X_clean), np.array(X_noisy), y

def preprocess_for_cnn(X_clean, X_noisy):
    """
    Preprocess density matrices for CNN input.
    
    Args:
        X_clean: Array of clean density matrices
        X_noisy: Array of noisy density matrices
    
    Returns:
        X: Input data (noisy density matrices)
        y: Target data (clean density matrices)
    """
    # Create 4-channel inputs: [real_noisy, imag_noisy, real_clean, imag_clean]
    # But we'll split them into X and y
    
    # For X (inputs): [real_noisy, imag_noisy]
    X = np.zeros((X_noisy.shape[0], X_noisy.shape[1], X_noisy.shape[2], 2))
    X[:, :, :, 0] = np.real(X_noisy)
    X[:, :, :, 1] = np.imag(X_noisy)
    
    # For y (targets): [real_clean, imag_clean]
    y = np.zeros((X_clean.shape[0], X_clean.shape[1], X_clean.shape[2], 2))
    y[:, :, :, 0] = np.real(X_clean)
    y[:, :, :, 1] = np.imag(X_clean)
    
    return X, y

def calculate_fidelity(rho, sigma):
    """
    Calculate fidelity between two density matrices.
    
    Args:
        rho: First density matrix
        sigma: Second density matrix
    
    Returns:
        Fidelity between the two density matrices
    """
    # Convert to numpy arrays if they're not already
    rho = np.array(rho)
    sigma = np.array(sigma)
    
    # Calculate the square root of rho
    sqrt_rho = scipy.linalg.sqrtm(rho)
    
    # Calculate the fidelity
    product = np.matmul(sqrt_rho, np.matmul(sigma, sqrt_rho))
    sqrt_product = scipy.linalg.sqrtm(product)
    
    # The trace of the square root is the fidelity
    fidelity = np.abs(np.trace(sqrt_product))
    
    return fidelity