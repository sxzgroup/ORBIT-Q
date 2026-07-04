// ORBIT-Q Webpage Interactive Script

// === Task Metadata Database ===
const tasksData = [
    {
        id: "01",
        title: "MPS Input and Circuit Refinement",
        workflow: "Matrix-product-state (MPS) input followed by variational circuit refinement.",
        objective: "Initialize the quantum simulator with a pre-contracted matrix product state representation and optimize a parameterized ansatz layout to match a target quantum phase profile under a strict energy minimization threshold.",
        category: "MPS, VQE"
    },
    {
        id: "02",
        title: "Energy and Entanglement Constraint",
        workflow: "Variational energy optimization with entanglement-profile constraints.",
        objective: "Perform variational quantum eigensolver optimization while introducing a penalty term targeting the local half-chain Von Neumann entanglement entropy, ensuring the resulting state is weakly entangled.",
        category: "VQE, Entanglement"
    },
    {
        id: "03",
        title: "Probability-Aware Cooling",
        workflow: "Probability-aware post-selected cooling with explicit success-rate tracking.",
        objective: "Iteratively cool a thermal-state mixture using mid-circuit measurement-feedback or sequential post-selection blocks, maximizing cooling efficiency while keeping track of the joint probability threshold.",
        category: "Post-selection"
    },
    {
        id: "04",
        title: "Kraus-Channel Calibration",
        workflow: "Trainable Kraus-channel calibration from multi-circuit data.",
        objective: "Fit the parameters of a set of local quantum noise operators (Kraus matrices) by backpropagating through circuit execution outputs compared against experimental observation distributions.",
        category: "Differentiable Noise"
    },
    {
        id: "05",
        title: "Imaginary Time Evolution",
        workflow: "Variational non-unitary imaginary time evolution.",
        objective: "Implement a non-unitary imaginary time operator on a quantum state using variational ansatz representations and auxiliary-qubit measurements to update parameters via the McLachlan variational principle.",
        category: "Variational Dynamics"
    },
    {
        id: "06",
        title: "Digital-Analog Optimization",
        workflow: "Digital-analog hybrid variational optimization.",
        objective: "Optimize hybrid parameters representing both discrete single-qubit gate phases (digital) and long continuous multi-qubit Hamiltonian interaction times (analog) using differentiable XLA compilation pathways.",
        category: "Digital-Analog"
    },
    {
        id: "07",
        title: "Measurement-Feedback Ground State",
        workflow: "Measurement-feedback variational optimization for ground states.",
        objective: "Evaluate the ground state of a spin Hamiltonian using a measurement-feedback loop, where ancilla measurements dynamically select subsequent gate parameters in the circuit.",
        category: "Feedback Control"
    },
    {
        id: "08",
        title: "2D Circuit Sampling",
        workflow: "Sampling from a 7 by 7 two-dimensional circuit.",
        objective: "Generate bitstring sampling samples from a 2D grid structure (7x7 qubits) which requires choosing an optimal tensor network contraction path (e.g. OMECO/Cotengra) without exploding memory limits.",
        category: "Tensor Networks"
    },
    {
        id: "09",
        title: "512-Qubit Local Observable",
        workflow: "Local-observable optimization in a 512-qubit shallow circuit.",
        objective: "Evaluate local observables on a large 512-qubit shallow circuit, exploiting localized causal light-cone properties to avoid dense statevector updates.",
        category: "Large-Scale Circuit"
    },
    {
        id: "10",
        title: "Nonlocal Multi-Qubit Optimization",
        workflow: "Variational optimization with large nonlocal multi-qubit gates.",
        objective: "Train a variational circuit containing large multi-qubit gate interactions (such as controlled multi-qubit Z gates) that require specialized tensor decompositions or dense matrix scaling policies.",
        category: "Nonlocal Gates"
    },
    {
        id: "11",
        title: "Haldane Spin-1 Verification",
        workflow: "Spin-1 Haldane-chain state preparation and string-order verification.",
        objective: "Prepare the symmetry-protected topological ground state of the spin-1 Haldane chain on a qutrit simulator and verify non-zero string order parameters using framework-native projection operations.",
        category: "Qutrit, SPT Phase"
    },
    {
        id: "12",
        title: "MPS Overlap Optimization",
        workflow: "Optimization of variational circuit overlap with an MPS target.",
        objective: "Optimize a quantum circuit's parameters to maximize its output state's overlap with an external MPS representation, requiring direct contraction between the circuit tensor network and MPS matrix cores.",
        category: "MPS, Overlap"
    }
];

// === Leaderboard Summary Database ===
const axisData = {
    framework: [
        { name: "TensorCircuit-NG", model: "Codex (GPT-5.5)", passes: 10, failures: 2, wall: "2h 05m 12s", efficiency: "2.2x", cost: "$17.99", costPerPass: "$1.80" },
        { name: "PennyLane", model: "Codex (GPT-5.5)", passes: 8, failures: 4, wall: "2h 39m 35s", efficiency: "4.5x", cost: "$23.04", costPerPass: "$2.88" },
        { name: "TorchQuantum", model: "Codex (GPT-5.5)", passes: 4, failures: 8, wall: "2h 18m 30s", efficiency: "7.3x", cost: "$24.91", costPerPass: "$6.23" },
        { name: "MindQuantum", model: "Codex (GPT-5.5)", passes: 4, failures: 8, wall: "2h 42m 42s", efficiency: "10.1x", cost: "$29.03", costPerPass: "$7.26" }
    ],
    agent: [
        { name: "Codex (GPT-5.5)", passes: 10, failures: 2, wall: "2h 05m 12s", efficiency: "2.3x", cost: "$17.99", costPerPass: "$1.80" },
        { name: "Codex (GPT-5.5*)", passes: 10, failures: 2, wall: "1h 29m 03s", efficiency: "2.2x", cost: "$13.29", costPerPass: "$1.33" },
        { name: "Claude Code (Opus-4.8)", passes: 9, failures: 3, wall: "2h 53m 02s", efficiency: "2.9x", cost: "$17.52", costPerPass: "$1.95" },
        { name: "Claude Code (Sonnet-4.6)", passes: 7, failures: 5, wall: "3h 47m 24s", efficiency: "7.2x", cost: "$14.55", costPerPass: "$2.08" },
        { name: "Claude Code (GLM-5.2)", passes: 6, failures: 6, wall: "3h 50m 11s", efficiency: "6.6x", cost: "$12.98", costPerPass: "$2.16" }
    ]
};

// === Task-by-Task Details Database ===
const detailsData = {
    // Framework Axis
    tensorcircuit: [
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "28m 35.5s", tokens: "4,196,140", cost: "$3.59", notes: "Agent timed out before producing a scorable artifact" },
        { task: "02", pass: true, runtime: "9.90s", ref: "2.87s", wall: "5m 53.3s", tokens: "1,031,334", cost: "$1.39", notes: "" },
        { task: "03", pass: true, runtime: "5.99s", ref: "2.46s", wall: "3m 07.1s", tokens: "494,792", cost: "$1.02", notes: "" },
        { task: "04", pass: true, runtime: "15.19s", ref: "11.83s", wall: "15m 29.1s", tokens: "1,390,826", cost: "$1.63", notes: "" },
        { task: "05", pass: true, runtime: "51.56s", ref: "45.50s", wall: "15m 46.5s", tokens: "1,200,716", cost: "$1.69", notes: "" },
        { task: "06", pass: true, runtime: "1m 40.3s", ref: "26.83s", wall: "9m 44.9s", tokens: "1,271,536", cost: "$1.57", notes: "" },
        { task: "07", pass: true, runtime: "1m 38.9s", ref: "1m 03.8s", wall: "7m 38.6s", tokens: "502,176", cost: "$1.05", notes: "" },
        { task: "08", pass: false, runtime: "5.43s", ref: "25.05s", wall: "11m 20.4s", tokens: "1,422,944", cost: "$1.92", notes: "Functional pass, audit fail: custom contraction/sampling bypassed TC" },
        { task: "09", pass: true, runtime: "35.61s", ref: "13.74s", wall: "4m 37.4s", tokens: "277,267", cost: "$0.71", notes: "" },
        { task: "10", pass: true, runtime: "1m 02.8s", ref: "12.44s", wall: "6m 50.6s", tokens: "460,817", cost: "$0.75", notes: "" },
        { task: "11", pass: true, runtime: "1m 32.7s", ref: "1m 08.1s", wall: "9m 17.0s", tokens: "1,119,151", cost: "$1.49", notes: "" },
        { task: "12", pass: true, runtime: "12.74s", ref: "6.12s", wall: "6m 51.9s", tokens: "873,992", cost: "$1.18", notes: "" }
    ],
    pennylane: [
        { task: "01", pass: false, runtime: "3m 03.9s", ref: "27.22s", wall: "19m 44.2s", tokens: "2,361,150", cost: "$3.02", notes: "Audit fail: sparse/partial optimization of declared ansatz" },
        { task: "02", pass: true, runtime: "9.95s", ref: "2.87s", wall: "3m 13.0s", tokens: "207,104", cost: "$0.67", notes: "" },
        { task: "03", pass: true, runtime: "38.95s", ref: "2.46s", wall: "4m 11.6s", tokens: "608,661", cost: "$0.95", notes: "" },
        { task: "04", pass: false, runtime: "8.12s", ref: "11.83s", wall: "7m 49.3s", tokens: "579,648", cost: "$1.30", notes: "Audit fail: hand-written analytic simulator replaces PL noisy circuit" },
        { task: "05", pass: true, runtime: "2m 11.8s", ref: "45.50s", wall: "10m 55.1s", tokens: "895,907", cost: "$1.22", notes: "" },
        { task: "06", pass: true, runtime: "1m 39.7s", ref: "26.83s", wall: "8m 25.0s", tokens: "1,235,168", cost: "$1.50", notes: "" },
        { task: "07", pass: false, runtime: "5.54s", ref: "1m 03.8s", wall: "10m 14.4s", tokens: "1,281,060", cost: "$1.83", notes: "Audit fail: trains a surrogate objective" },
        { task: "08", pass: true, runtime: "1m 00.0s", ref: "25.05s", wall: "17m 20.5s", tokens: "1,805,137", cost: "$2.52", notes: "" },
        { task: "09", pass: true, runtime: "2m 41.0s", ref: "13.74s", wall: "29m 26.7s", tokens: "2,412,391", cost: "$3.61", notes: "" },
        { task: "10", pass: true, runtime: "3m 52.9s", ref: "12.44s", wall: "18m 30.7s", tokens: "1,716,467", cost: "$2.13", notes: "" },
        { task: "11", pass: true, runtime: "2m 49.1s", ref: "1m 08.1s", wall: "15m 59.9s", tokens: "2,165,471", cost: "$2.47", notes: "" },
        { task: "12", pass: false, runtime: "3.25s", ref: "6.12s", wall: "13m 45.3s", tokens: "1,311,564", cost: "$1.82", notes: "Audit fail: JAX contraction bypassing PennyLane" }
    ],
    torchquantum: [
        { task: "01", pass: false, runtime: "3.90s", ref: "27.22s", wall: "12m 07.3s", tokens: "2,643,499", cost: "$3.10", notes: "Audit fail: local-RDM surrogate replaces full refinement circuit" },
        { task: "02", pass: true, runtime: "27.04s", ref: "2.87s", wall: "5m 50.0s", tokens: "859,854", cost: "$1.45", notes: "" },
        { task: "03", pass: true, runtime: "21.07s", ref: "2.46s", wall: "5m 07.7s", tokens: "556,044", cost: "$0.98", notes: "" },
        { task: "04", pass: false, runtime: "5.38s", ref: "11.83s", wall: "11m 00.0s", tokens: "2,183,300", cost: "$2.62", notes: "Audit fail: hand-written Kraus bypassing TQ" },
        { task: "05", pass: true, runtime: "1m 44.9s", ref: "45.50s", wall: "9m 46.6s", tokens: "1,773,601", cost: "$2.10", notes: "" },
        { task: "06", pass: true, runtime: "1m 43.6s", ref: "26.83s", wall: "11m 24.7s", tokens: "1,748,405", cost: "$2.14", notes: "" },
        { task: "07", pass: false, runtime: "3.37s", ref: "1m 03.8s", wall: "12m 31.5s", tokens: "1,670,437", cost: "$1.95", notes: "Audit fail: 8-qubit surrogate omits ancilla and feedback" },
        { task: "08", pass: false, runtime: "8.38s", ref: "25.05s", wall: "13m 00.2s", tokens: "1,283,844", cost: "$1.86", notes: "Manual-review fail: custom TN/MPS sampler bypassing TQ" },
        { task: "09", pass: false, runtime: "7.85s", ref: "13.74s", wall: "14m 19.2s", tokens: "2,343,141", cost: "$2.55", notes: "Manual-review fail: local-cone contraction bypassing TQ" },
        { task: "10", pass: false, runtime: "3m 56.2s", ref: "12.44s", wall: "17m 55.7s", tokens: "1,768,361", cost: "$2.48", notes: "Audit fail: CMZ applied by raw state-tensor phases" },
        { task: "11", pass: false, runtime: "2m 43.4s", ref: "1m 08.1s", wall: "14m 48.3s", tokens: "1,407,749", cost: "$1.54", notes: "Audit fail: qutrit rotation order mismatch" },
        { task: "12", pass: false, runtime: "40.87s", ref: "6.12s", wall: "10m 39.3s", tokens: "2,060,407", cost: "$2.14", notes: "Manual-review fail: custom MPS contraction bypassing TQ" }
    ],
    mindquantum: [
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "4m 16.0s", tokens: "795,609", cost: "$1.54", notes: "Obstruction: mqmps cannot support the required external-MPS workflow" },
        { task: "02", pass: true, runtime: "9.94s", ref: "2.87s", wall: "4m 48.7s", tokens: "551,949", cost: "$1.17", notes: "" },
        { task: "03", pass: true, runtime: "53.66s", ref: "2.46s", wall: "9m 55.1s", tokens: "1,451,151", cost: "$2.14", notes: "" },
        { task: "04", pass: false, runtime: "7.67s", ref: "11.83s", wall: "6m 46.3s", tokens: "656,613", cost: "$1.53", notes: "Audit fail: hand-written NumPy/Kraus transfer simulation" },
        { task: "05", pass: false, runtime: "1m 28.3s", ref: "45.50s", wall: "25m 43.1s", tokens: "5,073,450", cost: "$5.54", notes: "Static/semantic failure: full NumPy statevector simulator, no MQ import" },
        { task: "06", pass: false, runtime: "2m 24.7s", ref: "26.83s", wall: "26m 13.6s", tokens: "2,091,062", cost: "$2.58", notes: "Audit fail: clipped parameters and finite-difference gradients" },
        { task: "07", pass: false, runtime: "1m 18.6s", ref: "1m 03.8s", wall: "15m 25.1s", tokens: "1,683,461", cost: "$1.93", notes: "Audit fail: surrogate gradient replaces measurement-feedback objective" },
        { task: "08", pass: false, runtime: "1.19s", ref: "25.05s", wall: "5m 38.6s", tokens: "1,158,466", cost: "$1.84", notes: "Audit fail: mqmps MPS route replaces 2D TN contraction" },
        { task: "09", pass: true, runtime: "4m 49.4s", ref: "13.74s", wall: "28m 45.1s", tokens: "2,566,917", cost: "$2.99", notes: "" },
        { task: "10", pass: true, runtime: "1m 21.3s", ref: "12.44s", wall: "6m 59.3s", tokens: "446,389", cost: "$0.68", notes: "" },
        { task: "11", pass: false, runtime: "n/a", ref: "1m 08.1s", wall: "19m 35.1s", tokens: "4,734,997", cost: "$4.69", notes: "Missing solution artifact" },
        { task: "12", pass: false, runtime: "1m 08.3s", ref: "6.12s", wall: "8m 36.4s", tokens: "1,718,614", cost: "$2.40", notes: "Manual-review fail: custom NumPy TN overlap/gradient bypassing MQ" }
    ],

    // Agent Axis (Fixed TC Framework)
    "codex": [
        // Already defined under tensorcircuit
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "28m 35.5s", tokens: "4,196,140", cost: "$3.59", notes: "Agent timed out before producing a scorable artifact" },
        { task: "02", pass: true, runtime: "9.90s", ref: "2.87s", wall: "5m 53.3s", tokens: "1,031,334", cost: "$1.39", notes: "" },
        { task: "03", pass: true, runtime: "5.99s", ref: "2.46s", wall: "3m 07.1s", tokens: "494,792", cost: "$1.02", notes: "" },
        { task: "04", pass: true, runtime: "15.19s", ref: "11.83s", wall: "15m 29.1s", tokens: "1,390,826", cost: "$1.63", notes: "" },
        { task: "05", pass: true, runtime: "51.56s", ref: "45.50s", wall: "15m 46.5s", tokens: "1,200,716", cost: "$1.69", notes: "" },
        { task: "06", pass: true, runtime: "1m 40.3s", ref: "26.83s", wall: "9m 44.9s", tokens: "1,271,536", cost: "$1.57", notes: "" },
        { task: "07", pass: true, runtime: "1m 38.9s", ref: "1m 03.8s", wall: "7m 38.6s", tokens: "502,176", cost: "$1.05", notes: "" },
        { task: "08", pass: false, runtime: "5.43s", ref: "25.05s", wall: "11m 20.4s", tokens: "1,422,944", cost: "$1.92", notes: "Functional pass, audit fail: custom contraction/sampling bypassed TC" },
        { task: "09", pass: true, runtime: "35.61s", ref: "13.74s", wall: "4m 37.4s", tokens: "277,267", cost: "$0.71", notes: "" },
        { task: "10", pass: true, runtime: "1m 02.8s", ref: "12.44s", wall: "6m 50.6s", tokens: "460,817", cost: "$0.75", notes: "" },
        { task: "11", pass: true, runtime: "1m 32.7s", ref: "1m 08.1s", wall: "9m 17.0s", tokens: "1,119,151", cost: "$1.49", notes: "" },
        { task: "12", pass: true, runtime: "12.74s", ref: "6.12s", wall: "6m 51.9s", tokens: "873,992", cost: "$1.18", notes: "" }
    ],
    "codex-plus": [
        { task: "01", pass: false, runtime: "0.36s", ref: "27.22s", wall: "12m 58.3s", tokens: "1,731,676", cost: "$2.32", notes: "Functional pass, audit fail: simplified local-rotation model replaces brickwork VQE" },
        { task: "02", pass: true, runtime: "6.35s", ref: "2.87s", wall: "4m 44.2s", tokens: "481,893", cost: "$0.85", notes: "" },
        { task: "03", pass: true, runtime: "8.38s", ref: "2.46s", wall: "4m 27.7s", tokens: "449,821", cost: "$0.86", notes: "" },
        { task: "04", pass: true, runtime: "1m 26.6s", ref: "11.83s", wall: "9m 53.9s", tokens: "596,821", cost: "$0.98", notes: "" },
        { task: "05", pass: true, runtime: "1m 08.5s", ref: "45.50s", wall: "10m 25.3s", tokens: "960,110", cost: "$1.42", notes: "" },
        { task: "06", pass: true, runtime: "31.38s", ref: "26.83s", wall: "3m 38.4s", tokens: "330,129", cost: "$0.83", notes: "" },
        { task: "07", pass: true, runtime: "1m 30.7s", ref: "1m 03.8s", wall: "4m 21.7s", tokens: "339,750", cost: "$0.62", notes: "" },
        { task: "08", pass: false, runtime: "8.86s", ref: "25.05s", wall: "10m 47.3s", tokens: "928,227", cost: "$1.75", notes: "Human-review failure: custom NumPy column-transfer contraction/sampling bypassed TC" },
        { task: "09", pass: true, runtime: "35.38s", ref: "13.74s", wall: "3m 40.8s", tokens: "224,685", cost: "$0.50", notes: "" },
        { task: "10", pass: true, runtime: "34.18s", ref: "12.44s", wall: "3m 33.6s", tokens: "428,818", cost: "$0.68", notes: "" },
        { task: "11", pass: true, runtime: "1m 38.7s", ref: "1m 08.1s", wall: "14m 10.2s", tokens: "791,903", cost: "$1.27", notes: "" },
        { task: "12", pass: true, runtime: "11.00s", ref: "6.12s", wall: "6m 22.0s", tokens: "941,744", cost: "$1.21", notes: "" }
    ],
    "opus": [
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "21m 45.5s", tokens: "2,784,543", cost: "$2.28", notes: "Submitted artifact timed out during evaluation" },
        { task: "02", pass: true, runtime: "8.50s", ref: "2.87s", wall: "6m 24.2s", tokens: "810,473", cost: "$1.29", notes: "" },
        { task: "03", pass: true, runtime: "4.54s", ref: "2.46s", wall: "23m 35.1s", tokens: "1,079,876", cost: "$1.51", notes: "" },
        { task: "04", pass: true, runtime: "2m 47.0s", ref: "11.83s", wall: "30m 00.0s", tokens: "1,524,732", cost: "$2.62", notes: "" },
        { task: "05", pass: true, runtime: "48.36s", ref: "45.50s", wall: "9m 18.2s", tokens: "779,913", cost: "$1.34", notes: "" },
        { task: "06", pass: true, runtime: "1m 47.1s", ref: "26.83s", wall: "9m 13.1s", tokens: "743,764", cost: "$1.37", notes: "" },
        { task: "07", pass: true, runtime: "1m 21.4s", ref: "1m 03.8s", wall: "30m 00.0s", tokens: "1,156,796", cost: "$2.08", notes: "" },
        { task: "08", pass: false, runtime: "n/a", ref: "25.05s", wall: "2m 31.8s", tokens: "194,149", cost: "$0.51", notes: "Refusal: Cyber-safeguard security block during local TC CLI testing" },
        { task: "09", pass: true, runtime: "34.41s", ref: "13.74s", wall: "8m 50.3s", tokens: "551,429", cost: "$0.97", notes: "" },
        { task: "10", pass: true, runtime: "1m 33.3s", ref: "12.44s", wall: "9m 21.0s", tokens: "741,753", cost: "$0.78", notes: "" },
        { task: "11", pass: true, runtime: "2m 02.0s", ref: "1m 08.1s", wall: "19m 10.2s", tokens: "1,523,481", cost: "$1.96", notes: "" },
        { task: "12", pass: false, runtime: "n/a", ref: "6.12s", wall: "2m 53.0s", tokens: "312,849", cost: "$0.81", notes: "Refusal: Cyber-safeguard security block during local TC CLI testing" }
    ],
    "sonnet": [
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "30m 00.0s", tokens: "4,351,656", cost: "$2.01", notes: "Submitted artifact timed out during functional evaluation" },
        { task: "02", pass: true, runtime: "1m 21.6s", ref: "2.87s", wall: "30m 00.0s", tokens: "2,177,375", cost: "$2.05", notes: "" },
        { task: "03", pass: true, runtime: "25.67s", ref: "2.46s", wall: "24m 56.9s", tokens: "2,346,526", cost: "$1.59", notes: "" },
        { task: "04", pass: false, runtime: "n/a", ref: "11.83s", wall: "30m 00.1s", tokens: "2,772,013", cost: "$2.31", notes: "Agent timed out before producing a solution artifact" },
        { task: "05", pass: true, runtime: "1m 12.0s", ref: "45.50s", wall: "11m 43.6s", tokens: "1,369,687", cost: "$1.15", notes: "" },
        { task: "06", pass: false, runtime: "n/a", ref: "26.83s", wall: "5m 37.5s", tokens: "610,159", cost: "$0.44", notes: "Exception: PauliStringSum2COO received string Pauli labels under JAX" },
        { task: "07", pass: true, runtime: "2m 17.2s", ref: "1m 03.8s", wall: "18m 53.8s", tokens: "1,591,490", cost: "$0.79", notes: "" },
        { task: "08", pass: false, runtime: "2.87s", ref: "25.05s", wall: "19m 09.5s", tokens: "1,311,477", cost: "$1.13", notes: "Audit fail: handwritten 2D tensor-network sampling bypassing TC" },
        { task: "09", pass: true, runtime: "1m 22.9s", ref: "13.74s", wall: "12m 56.3s", tokens: "1,191,784", cost: "$0.78", notes: "" },
        { task: "10", pass: true, runtime: "9m 27.7s", ref: "12.44s", wall: "7m 40.0s", tokens: "959,901", cost: "$0.53", notes: "" },
        { task: "11", pass: true, runtime: "4m 16.9s", ref: "1m 08.1s", wall: "22m 24.4s", tokens: "1,914,092", cost: "$0.88", notes: "" },
        { task: "12", pass: false, runtime: "n/a", ref: "6.12s", wall: "14m 02.1s", tokens: "672,635", cost: "$0.89", notes: "No solution artifact was produced" }
    ],
    "glm": [
        { task: "01", pass: false, runtime: "n/a", ref: "27.22s", wall: "30m 00.1s", tokens: "3,680,174", cost: "$3.19", notes: "Agent timed out before producing a solution artifact" },
        { task: "02", pass: true, runtime: "16.89s", ref: "2.87s", wall: "5m 02.0s", tokens: "570,510", cost: "$0.38", notes: "" },
        { task: "03", pass: true, runtime: "1m 56.7s", ref: "2.46s", wall: "30m 00.1s", tokens: "3,518,912", cost: "$2.48", notes: "" },
        { task: "04", pass: true, runtime: "1m 23.8s", ref: "11.83s", wall: "24m 09.8s", tokens: "2,540,761", cost: "$1.07", notes: "" },
        { task: "05", pass: true, runtime: "1m 32.3s", ref: "45.50s", wall: "11m 59.3s", tokens: "592,114", cost: "$0.37", notes: "" },
        { task: "06", pass: true, runtime: "2m 32.8s", ref: "26.83s", wall: "25m 24.0s", tokens: "2,204,624", cost: "$1.42", notes: "" },
        { task: "07", pass: false, runtime: "n/a", ref: "1m 03.8s", wall: "24m 34.1s", tokens: "1,225,285", cost: "$0.73", notes: "Agent exited 137 after heavy TC DMCircuit running; no scorable artifact" },
        { task: "08", pass: false, runtime: "n/a", ref: "25.05s", wall: "12m 00.4s", tokens: "555,137", cost: "$0.35", notes: "Agent exited 137 after heavy TC contraction/sampling; no scorable artifact" },
        { task: "09", pass: false, runtime: "8.47s", ref: "13.74s", wall: "10m 48.1s", tokens: "600,180", cost: "$0.34", notes: "Audit fail: omitted required plus-state initialization" },
        { task: "10", pass: true, runtime: "45.67s", ref: "12.44s", wall: "17m 19.0s", tokens: "1,738,141", cost: "$0.79", notes: "" },
        { task: "11", pass: false, runtime: "n/a", ref: "1m 08.1s", wall: "21m 33.5s", tokens: "1,530,951", cost: "$0.79", notes: "Verifier evaluation process killed from resource-heavy qutrit execution" },
        { task: "12", pass: false, runtime: "2m 29.7s", ref: "6.12s", wall: "17m 20.7s", tokens: "2,318,388", cost: "$1.07", notes: "Audit fail: mismatch implementation from the problem" }
    ]
};

// === DOM Manipulation and UI Setup ===
document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Paper Figure Tabs ---
    const figTabBtns = document.querySelectorAll(".fig-tab-btn");
    const figPanels = document.querySelectorAll(".fig-content-panel");
    
    figTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetFig = btn.getAttribute("data-fig");
            
            figTabBtns.forEach(b => b.classList.remove("active"));
            figPanels.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`${targetFig}-panel`).classList.add("active");
        });
    });
    
    // --- 2. Leaderboard Axis Tabs ---
    const leadTabBtns = document.querySelectorAll(".lead-tab-btn");
    const axisViews = document.querySelectorAll(".axis-view");
    
    leadTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetAxis = btn.getAttribute("data-axis");
            
            leadTabBtns.forEach(b => b.classList.remove("active"));
            axisViews.forEach(v => v.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`${targetAxis}-view`).classList.add("active");
            
            renderLeaderboard(targetAxis);
        });
    });
    
    // --- 3. Leaderboard Selectors (Framework / Agent Axis Sub-tables) ---
    const setupSelectorListeners = (selectorClass, tableId, dataKeyAttr) => {
        const btns = document.querySelectorAll(`.${selectorClass} .selector-btn`);
        btns.forEach(btn => {
            btn.addEventListener("click", () => {
                const parent = btn.parentElement;
                parent.querySelectorAll(".selector-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                
                const targetKey = btn.getAttribute(dataKeyAttr);
                renderDetailsTable(tableId, targetKey);
            });
        });
    };
    
    setupSelectorListeners("framework-selectors", "framework-details-table", "data-fw");
    setupSelectorListeners("agent-selectors", "agent-details-table", "data-agent");
    
    // --- 4. Render Leaderboard and Sub-tables ---
    const renderLeaderboard = (axis) => {
        const data = axisData[axis];
        const tbody = document.querySelector(`#${axis}-table tbody`);
        const summaryContainer = document.getElementById(`${axis}-summary-cards`);
        
        tbody.innerHTML = "";
        summaryContainer.innerHTML = "";
        
        // Render Summary Cards
        if (axis === "framework") {
            const bestFw = data[0]; // TensorCircuit-NG has 10 passes
            createSummaryCard(summaryContainer, "Top Framework", bestFw.name, "by valid passes");
            createSummaryCard(summaryContainer, "Artifact Efficiency", "2.2x Slowdown", "TC vs 4.5x–10.1x for other frameworks");
            createSummaryCard(summaryContainer, "Success Rate Ceiling", "83.3%", "10 of 12 (TensorCircuit-NG)");
            createSummaryCard(summaryContainer, "Average Solving cost", "$1.80", "per valid TC solution");
        } else {
            const bestAgent = data[1]; // Codex Plus has 1.33 cost per pass
            createSummaryCard(summaryContainer, "Best Cost Efficiency", bestAgent.name, "$1.33 cost / pass");
            createSummaryCard(summaryContainer, "Top Agent Passes", "10 / 12", "Codex (GPT-5.5 & GPT-5.5*)");
            createSummaryCard(summaryContainer, "Best Artifact Efficiency", "2.2x Slowdown", "Codex + GPT-5.5 produces most efficient code");
        }
        
        // Render Table Rows
        data.forEach(row => {
            const tr = document.createElement("tr");
            
            if (axis === "framework") {
                tr.innerHTML = `
                    <td style="font-weight: 600; color: var(--text-white);">${row.name}</td>
                    <td style="color: var(--text-secondary);">${row.model}</td>
                    <td class="num-col" style="color: #10b981; font-weight: 600;">${row.passes}</td>
                    <td class="num-col" style="color: #ef4444;">${row.failures}</td>
                    <td class="num-col">${row.wall}</td>
                    <td class="num-col" style="font-weight: 600; color: var(--text-white);">${row.efficiency}</td>
                    <td class="num-col">${row.cost}</td>
                    <td class="num-col" style="color: var(--accent); font-weight: 500;">${row.costPerPass}</td>
                `;
            } else {
                tr.innerHTML = `
                    <td style="font-weight: 600; color: var(--text-white);">${row.name}</td>
                    <td class="num-col" style="color: #10b981; font-weight: 600;">${row.passes}</td>
                    <td class="num-col" style="color: #ef4444;">${row.failures}</td>
                    <td class="num-col">${row.wall}</td>
                    <td class="num-col" style="font-weight: 600; color: var(--text-white);">${row.efficiency}</td>
                    <td class="num-col">${row.cost}</td>
                    <td class="num-col" style="color: var(--accent); font-weight: 500;">${row.costPerPass}</td>
                `;
            }
            tbody.appendChild(tr);
        });
    };
    
    const createSummaryCard = (container, label, value, sublabel) => {
        const card = document.createElement("div");
        card.className = "summary-card";
        card.innerHTML = `
            <h4>${label}</h4>
            <div class="metric-val">${value}</div>
            <div class="metric-lbl">${sublabel}</div>
        `;
        container.appendChild(card);
    };
    
    const renderDetailsTable = (tableId, key) => {
        const data = detailsData[key];
        const tbody = document.querySelector(`#${tableId} tbody`);
        tbody.innerHTML = "";
        
        data.forEach(row => {
            const tr = document.createElement("tr");
            
            // Format Pass Badge
            const passBadge = row.pass 
                ? `<span class="badge-pass">PASS</span>` 
                : `<span class="badge-fail">FAIL</span>`;
                
            // Format Slowdown vs Ref (e.g. Solution / Ref)
            let speedCol = "";
            if (!row.pass || row.runtime === "n/a") {
                speedCol = `<span style="color:var(--text-muted);">n/a</span>`;
            } else {
                // If we want to show solution vs reference, format it nicely
                speedCol = `<span style="font-weight:600; color:var(--text-white);">${row.runtime}</span>`;
            }
            
            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-weight: 600; color: var(--purple);">Task ${row.task}</td>
                <td>${passBadge}</td>
                <td class="num-col">${speedCol}</td>
                <td class="num-col"><span class="badge-ref">${row.ref}</span></td>
                <td class="num-col">${row.wall}</td>
                <td class="num-col">${row.tokens}</td>
                <td class="num-col">${row.cost}</td>
                <td style="font-size: 0.85rem; color: var(--text-secondary);">${row.notes || '<span style="color:var(--text-muted);">—</span>'}</td>
            `;
            tbody.appendChild(tr);
        });
    };
    
    // Initial Render
    renderLeaderboard("framework");
    renderDetailsTable("framework-details-table", "tensorcircuit");
    renderDetailsTable("agent-details-table", "codex");
    
    // --- 5. Task Explorer ---
    const taskGrid = document.getElementById("task-cards-grid");
    const detailPanel = document.getElementById("task-details-panel");
    const emptyState = document.getElementById("panel-empty");
    const contentState = document.getElementById("panel-content");
    
    // Render Task Cards
    tasksData.forEach(task => {
        const card = document.createElement("div");
        card.className = "task-card";
        card.setAttribute("data-task-id", task.id);
        card.innerHTML = `
            <span class="task-card-id">Challenge ${task.id}</span>
            <div class="task-card-title">${task.title}</div>
        `;
        taskGrid.appendChild(card);
        
        card.addEventListener("click", () => {
            // Toggle active card styling
            document.querySelectorAll(".task-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            
            // Populate Details Panel
            emptyState.classList.add("hidden");
            contentState.classList.remove("hidden");
            
            document.getElementById("detail-id").innerText = `Challenge ${task.id}`;
            document.getElementById("detail-title").innerText = task.title;
            document.getElementById("detail-workflow").innerText = task.workflow;
            document.getElementById("detail-objective").innerText = task.objective;
            
            // Set dynamic GitHub links
            document.getElementById("github-problem-link").href = `https://github.com/sxzgroup/ORBIT-Q/blob/main/tasks/challenge-${task.id}/instruction.md`;
            document.getElementById("github-solution-link").href = `https://github.com/sxzgroup/ORBIT-Q/tree/main/templates/challenge/tests/`;
        });
    });
    
    // --- 6. Quick Start Terminal Tab Switcher ---
    const termTabBtns = document.querySelectorAll(".term-tab-btn");
    const termContents = document.querySelectorAll(".term-content");
    
    termTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTerm = btn.getAttribute("data-term");
            
            termTabBtns.forEach(b => b.classList.remove("active"));
            termContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(`term-${targetTerm}-content`).classList.add("active");
        });
    });
    
    // --- 7. Copy Command Logic ---
    const copyToClipboard = (elementId, buttonId, successText) => {
        const btn = document.getElementById(buttonId);
        const codeElement = document.getElementById(elementId);
        
        btn.addEventListener("click", () => {
            let textToCopy = "";
            
            if (codeElement.tagName === "CODE" || codeElement.tagName === "PRE") {
                textToCopy = codeElement.innerText;
            } else {
                // If it's a terminal tab content, we collect the commands inside active tab
                const activeContent = codeElement.querySelector(".term-content.active");
                const commands = activeContent.querySelectorAll(".term-line .term-cmd");
                const textArr = [];
                commands.forEach(cmd => {
                    // Skip copy commands or comments
                    if (!cmd.innerText.startsWith("#")) {
                        textArr.push(cmd.innerText);
                    }
                });
                textToCopy = textArr.join("\n");
            }
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalHTML = btn.innerHTML;
                btn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    ${successText}
                `;
                btn.style.color = "var(--success)";
                btn.style.borderColor = "var(--success-border)";
                
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.style.color = "";
                    btn.style.borderColor = "";
                }, 2000);
            }).catch(err => {
                console.error("Failed to copy text: ", err);
            });
        });
    };
    
    copyToClipboard("term-bootstrap-content", "terminal-copy-btn", "Commands Copied!");
    copyToClipboard("bibtex-code", "citation-copy-btn", "BibTeX Copied!");
    
    // Intercept copy button click for terminal specifically to check active panel commands
    const terminalCopyBtn = document.getElementById("terminal-copy-btn");
    terminalCopyBtn.addEventListener("click", () => {
        const activeContent = document.querySelector(".term-content.active");
        const lines = activeContent.querySelectorAll(".term-line");
        const textArr = [];
        lines.forEach(line => {
            const cmd = line.querySelector(".term-cmd");
            if (cmd && !cmd.innerText.trim().startsWith("#")) {
                textArr.push(cmd.innerText.trim());
            }
        });
        
        navigator.clipboard.writeText(textArr.join("\n")).then(() => {
            const originalHTML = terminalCopyBtn.innerHTML;
            terminalCopyBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                Copied!
            `;
            terminalCopyBtn.style.color = "var(--success)";
            terminalCopyBtn.style.borderColor = "var(--success-border)";
            
            setTimeout(() => {
                terminalCopyBtn.innerHTML = originalHTML;
                terminalCopyBtn.style.color = "";
                terminalCopyBtn.style.borderColor = "";
            }, 2000);
        });
    });

    // --- 8. Theme Toggle Switch ---
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const sunIcon = themeToggleBtn.querySelector(".sun-icon");
    const moonIcon = themeToggleBtn.querySelector(".moon-icon");
    
    // Check saved preference, otherwise follow system (defaulting to light)
    const getSystemTheme = () => {
        if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            return "dark";
        }
        return "light";
    };
    
    const savedTheme = localStorage.getItem("theme") || getSystemTheme();
    document.documentElement.setAttribute("data-theme", savedTheme);
    
    if (savedTheme === "light") {
        sunIcon.classList.add("hidden");
        moonIcon.classList.remove("hidden");
    } else {
        sunIcon.classList.remove("hidden");
        moonIcon.classList.add("hidden");
    }
    
    themeToggleBtn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        let newTheme = "dark";
        
        if (currentTheme === "dark") {
            newTheme = "light";
            sunIcon.classList.add("hidden");
            moonIcon.classList.remove("hidden");
        } else {
            newTheme = "dark";
            sunIcon.classList.remove("hidden");
            moonIcon.classList.add("hidden");
        }
        
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });
});
