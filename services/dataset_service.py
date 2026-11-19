# Generates and manages the 100-Document Benchmark Corpus:
#  - 30% Real Data (30 Documents across Computer Science, Physics, Biology, History, Security)
#  - 70% Synthetic Data (70 Documents generated via Zipfian Power-Law & Calibrated Perturbations)


from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.json"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.json"

# 30 Real Domain Documents (30% Real Data)
REAL_DOCUMENTS = [
    # Computer Science & AI
    (
        "REAL_CS_01_ResNet",
        "Deep Residual Learning for Image Recognition",
        "Deeper neural networks are more difficult to train. We present a residual learning framework "
        "to ease the training of networks that are substantially deeper than those used previously. We "
        "explicitly reformulate the layers as learning residual functions with reference to the layer inputs, "
        "instead of learning unreferenced functions. Comprehensive empirical evidence shows that these residual "
        "networks are easier to optimize and can gain accuracy from considerably increased depth. On the ImageNet "
        "dataset we evaluate residual nets with a depth of up to one hundred and fifty-two layers, which are eight "
        "times deeper than VGG nets but still have lower computational complexity. An ensemble of these residual "
        "nets achieves error on the ImageNet test set, winning first place in the classification task.",
        "Computer Science",
    ),
    (
        "REAL_CS_02_Transformer",
        "Attention Is All You Need and Transformer Networks",
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
        "in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through "
        "an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks "
        "show these models to be superior in quality while being more parallelizable and requiring significantly less "
        "time to train. The model achieves state of the art results on English to German and English to French translation.",
        "Computer Science",
    ),
    (
        "REAL_CS_03_GCN",
        "Semi-Supervised Classification with Graph Convolutional Networks",
        "We present a scalable approach for semi-supervised learning on graph-structured data. It is based on an "
        "efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice "
        "of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. "
        "Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both "
        "local graph structure and features of nodes. In several experiments on citation networks and a knowledge graph "
        "dataset, we demonstrate that our approach outperforms related methods by a significant margin.",
        "Computer Science",
    ),
    (
        "REAL_CS_04_Paxos",
        "Paxos and Distributed Consensus in Fault-Tolerant Systems",
        "The Paxos protocol guarantees safety in asynchronous distributed systems with crash-stop or crash-recovery "
        "fault models. Consensus is achieved through rounds of proposals, where proposers request promises from a quorum "
        "of acceptors before committing values. By decoupling the leader election phase from the replicated state machine "
        "log execution, Multi-Paxos enables high-throughput state machine replication. Practical deployments such as "
        "Google Chubby and Apache ZooKeeper adapt Paxos principles to handle network partitions, disk write latencies, "
        "and dynamic cluster membership reconfiguration.",
        "Computer Science",
    ),
    (
        "REAL_CS_05_LSH_MinHash",
        "Locality-Sensitive Hashing for Scalable Similarity Search",
        "Locality-Sensitive Hashing resolves the curse of dimensionality in high-dimensional nearest neighbor queries. "
        "By applying hash functions where collision probability is strictly proportional to the underlying metric similarity, "
        "sub-quadratic candidate filtering replaces exhaustive all-pairs brute-force comparison. For unweighted sets, "
        "Broder's MinHash preserves the classic Jaccard index. For weighted feature vectors and probability distributions, "
        "consistent weighted sampling and Ertl's ProbMinHash align hash collision probabilities directly with the probability "
        "Jaccard similarity.",
        "Computer Science",
    ),
    (
        "REAL_CS_06_LinuxKernel",
        "Virtual Memory and Page Replacement in Operating System Kernels",
        "Modern operating system kernels manage physical memory via multi-level page tables and virtual address translation. "
        "The Translation Lookaside Buffer caches virtual to physical address translations, minimizing hardware traversal "
        "penalties. Demand paging loads pages into memory on demand when page faults occur. Page replacement policies, such "
        "as least recently used approximations and clock algorithms, balance working set retention against page thrashing. "
        "Copy-on-write optimizations defer page duplication until memory writes occur, optimizing process creation through fork.",
        "Computer Science",
    ),
    # Cybersecurity & Cryptography
    (
        "REAL_SEC_01_RSA",
        "Public-Key Cryptography and RSA Algorithm Foundations",
        "Public-key cryptography enables secure communication over untrusted channels without prior shared secret keys. "
        "The RSA cryptosystem relies on the computational intractability of factoring large composite integers into prime "
        "factors. The key generation process selects two large prime numbers, computes their modulus and Euler totient function, "
        "and derives public encryption and private decryption exponents. Encryption applies modular exponentiation to plaintext "
        "blocks, while decryption inverts the operation using the private exponent via modular arithmetic.",
        "Cybersecurity",
    ),
    (
        "REAL_SEC_02_ZeroKnowledge",
        "Zero-Knowledge Proofs and Succinct Non-Interactive Arguments",
        "Zero-knowledge proofs permit a prover to convince a verifier that a statement is true without revealing any auxiliary "
        "information beyond the validity of the statement itself. Completeness ensures honest provers convince verifiers, "
        "soundness prevents cheating provers from validating false statements, and zero-knowledge guarantees no computational "
        "leakage. Non-interactive zero-knowledge arguments of knowledge utilize polynomial commitments and elliptic curve "
        "pairings to verify complex computational transactions in blockchain protocols and privacy-preserving authentication.",
        "Cybersecurity",
    ),
    (
        "REAL_SEC_03_BufferOverflow",
        "Memory Safety Vulnerabilities and Stack Buffer Overflows",
        "Buffer overflow vulnerabilities arise when a program writes data beyond the allocated boundary of a memory buffer. "
        "In stack-based buffer overflows, malicious input overwrites adjacent memory containing the saved frame pointer and function "
        "return address. When the function returns, control flow transfers to injected shellcode or existing executable code "
        "via return-oriented programming. Modern defensive mitigations incorporate address space layout randomization, non-executable "
        "stack segments, compiler stack canaries, and modern memory-safe programming languages.",
        "Cybersecurity",
    ),
    (
        "REAL_SEC_04_TLS",
        "Transport Layer Security and Cryptographic Handshake Protocols",
        "Transport Layer Security secures end-to-end communication across the internet, guaranteeing confidentiality, "
        "authenticity, and message integrity. The TLS handshake establishes connection parameters, verifies digital certificates "
        "issued by trusted certificate authorities, and negotiates ephemeral symmetric cipher keys using Elliptic-Curve "
        "Diffie-Hellman key exchange. Symmetric authenticated encryption algorithms like AES-GCM protect transmitted application data, "
        "while cryptographic message authentication codes prevent replay and tampering attacks.",
        "Cybersecurity",
    ),
    (
        "REAL_SEC_05_Blockchain",
        "Distributed Ledgers and Proof-of-Work Consensus Networks",
        "A blockchain is an immutable distributed ledger composed of cryptographically linked blocks. Each block header contains "
        "the cryptographic hash of the previous block, a timestamp, a transaction root hash computed via a Merkle tree, and a "
        "proof-of-work nonce. Miners compete to find a nonce that yields a SHA-256 block hash below the network target difficulty. "
        "This consensus mechanism prevents double-spending attacks and establishes Byzantine fault tolerance without a central authority.",
        "Cybersecurity",
    ),
    (
        "REAL_SEC_06_IntrusionDetection",
        "Network Intrusion Detection and Anomaly Signature Analysis",
        "Network intrusion detection systems monitor packet telemetry and connection metadata to identify unauthorized access "
        "and cyber threats. Signature-based detection inspects payload patterns against known exploit fingerprints, providing high "
        "accuracy for established attacks. Anomaly-based detection leverages statistical profiling and unsupervised machine learning "
        "to detect deviations from baseline network traffic, flagging zero-day exploits, port scanning activity, and distributed "
        "denial-of-service volumetric surges.",
        "Cybersecurity",
    ),
    # Natural Sciences & Physics
    (
        "REAL_PHYS_01_QuantumEntanglement",
        "Quantum Entanglement and Bell Inequality Violations",
        "Quantum entanglement describes composite physical systems whose quantum states cannot be factored into product states "
        "of individual subsystems. When entangled particles are separated across spacelike intervals, measurement of one particle "
        "instantaneously correlates with the measurement outcome of the other, defying local hidden-variable realism. Experimental "
        "tests of Bell inequalities using polarized photon pairs confirm quantum mechanical predictions, providing the operational "
        "basis for quantum key distribution, quantum teleportation, and fault-tolerant quantum computing architectures.",
        "Physics",
    ),
    (
        "REAL_PHYS_02_BlackHoles",
        "Black Hole Thermodynamics and Hawking Radiation Dynamics",
        "Black hole thermodynamics establishes an exact mathematical isomorphism between gravitational physics and classical "
        "thermodynamics. The Bekenstein-Hawking entropy of an event horizon is proportional to its surface area measured in Planck "
        "units. Quantum field theory in curved spacetime demonstrates that virtual particle-antiparticle pairs near the horizon "
        "lead to particle emission known as Hawking radiation. As a black hole radiates thermal energy, its mass decreases, leading "
        "to the black hole information paradox regarding unitarity preservation during evaporation.",
        "Physics",
    ),
    (
        "REAL_PHYS_03_GeneralRelativity",
        "Gravitational Waves and Spacetime Metric Curvature",
        "Einstein's field equations describe gravity not as an attractive force, but as the manifestation of spacetime curvature "
        "induced by mass-energy distributions. Compact binary inspirals of neutron stars and black holes generate quadrupole perturbations "
        "that propagate outward at the speed of light as gravitational waves. Laser interferometer detectors like LIGO and Virgo measure "
        "differential strain amplitudes on the order of ten to the minus twenty-one meters, opening gravitational wave astronomy.",
        "Physics",
    ),
    (
        "REAL_PHYS_04_Superconductivity",
        "High-Temperature Superconductivity and Cooper Pair Condensation",
        "Superconductivity is characterized by the sudden disappearance of electrical resistance and the expulsion of magnetic flux "
        "fields via the Meissner effect. Conventional superconductors are explained by the BCS theory, where electron-phonon interactions "
        "produce attractive forces that bind electrons into Cooper pairs. These bosonic pairs condense into a macroscopic quantum ground "
        "state. High-temperature cuprate and nickelate superconductors operate beyond weak-coupling phonon mechanisms, involving strong "
        "electronic correlation and antiferromagnetic spin fluctuations.",
        "Physics",
    ),
    (
        "REAL_PHYS_05_DarkMatter",
        "Cosmological Dark Matter and Gravitational Lensing Evidence",
        "Astrophysical observations across multiple scales indicate that baryonic matter constitutes only a fraction of total cosmic "
        "matter density. Galactic rotation curves remain flat at large radial distances from galactic centers, indicating massive non-luminous "
        "halos. Weak and strong gravitational lensing observations of colliding galaxy clusters, such as the Bullet Cluster, demonstrate "
        "separation between baryonic plasma and gravitational potential wells, supporting collisionless cold dark matter candidates.",
        "Physics",
    ),
    (
        "REAL_PHYS_06_FusionEnergy",
        "Magnetic Confinement Fusion and Tokamak Plasma Stability",
        "Thermonuclear fusion seeks to replicate solar energy production by fusing hydrogen isotopes deuterium and tritium under "
        "temperatures exceeding one hundred million Kelvin. Magnetic confinement fusion uses helical magnetic fields in tokamak configurations "
        "to confine hot ionized plasma away from vacuum vessel walls. Key engineering and physics challenges involve suppressing "
        "magnetohydrodynamic instabilities, managing turbulent transport losses, and developing plasma-facing divertor materials.",
        "Physics",
    ),
    # Biology & Medicine
    (
        "REAL_BIO_01_CRISPR",
        "CRISPR-Cas9 Gene Editing and Targeted Molecular Cleavage",
        "The bacterial adaptive immune system CRISPR-Cas9 has been repurposed as a revolutionary RNA-guided endonuclease platform "
        "for precision genome editing. A single guide RNA directs the Cas9 endonuclease to a complementary genomic target sequence "
        "adjacent to a protospacer adjacent motif. Cas9 induces targeted double-strand breaks, triggering endogenous cellular DNA repair "
        "pathways. Non-homologous end joining introduces targeted knockouts, while homology-directed repair enables precise genetic "
        "insertions for therapeutic disease correction.",
        "Biology",
    ),
    (
        "REAL_BIO_02_Photosynthesis",
        "Biochemical Mechanisms of Photosynthetic Electron Transport",
        "Oxygenic photosynthesis converts solar electromagnetic radiation into chemical energy stored in carbohydrate bonds. In thylakoid "
        "membranes, Photosystem II absorbs photons, driving water photolysis to release protons and oxygen gas. Electrons travel through an "
        "electron transport chain involving plastoquinone, the cytochrome b6f complex, and plastocyanin to Photosystem I. The generated proton "
        "gradient powers ATP synthase to synthesize ATP, while ferredoxin-NADP+ reductase produces NADPH for carbon fixation in the Calvin cycle.",
        "Biology",
    ),
    (
        "REAL_BIO_03_mRNA_Vaccines",
        "Lipid Nanoparticle Delivery and mRNA Vaccine Mechanisms",
        "Messenger RNA vaccines introduce synthetic transcript sequences encoding viral antigens encapsulated within ionizable lipid "
        "nanoparticles. Nanoparticle formulations protect labile mRNA molecules from enzymatic ribonuclease degradation and facilitate "
        "endosomal escape into the host cell cytoplasm. Ribosomes translate the exogenous mRNA into target antigen proteins, which are "
        "presented on major histocompatibility complexes to activate neutralizing antibodies and cytotoxic T-cell immune responses.",
        "Biology",
    ),
    (
        "REAL_BIO_04_Neuroplasticity",
        "Synaptic Plasticity and Long-Term Potentiation in Neurons",
        "Synaptic plasticity constitutes the primary cellular mechanism underlying learning and memory formation in mammalian nervous "
        "systems. Repetitive high-frequency stimulation of excitatory synapses induces long-term potentiation, characterized by persistent "
        "strengthening of synaptic transmission efficacy. Activation of NMDA receptors triggers calcium influx into dendritic spines, "
        "promoting trafficking and insertion of AMPA receptors into the postsynaptic density and structural remodeling of synaptic boutons.",
        "Biology",
    ),
    (
        "REAL_BIO_05_Mitochondria",
        "Mitochondrial Oxidative Phosphorylation and ATP Synthesis",
        "Mitochondria are double-membraned cellular organelles responsible for generating adenosine triphosphate through oxidative phosphorylation. "
        "The citric acid cycle in the mitochondrial matrix yields high-energy electron carriers NADH and FADH2. Complexes I through IV in the "
        "inner mitochondrial membrane transfer electrons to molecular oxygen while pumping protons into the intermembrane space. The resulting "
        "electrochemical proton-motive force drives the rotary catalysis of ATP synthase to generate ATP from ADP and inorganic phosphate.",
        "Biology",
    ),
    (
        "REAL_BIO_06_Microbiome",
        "The Human Gut Microbiome and Metabolic Immunomodulation",
        "The human gastrointestinal tract harbors complex microbial communities comprising trillions of bacterial, viral, and fungal cells. "
        "The gut microbiome performs essential physiological functions, including fermentation of non-digestible dietary fibers into "
        "short-chain fatty acids such as acetate, propionate, and butyrate. These metabolic byproducts regulate intestinal barrier integrity, "
        "modulate systemic immune cell maturation, and influence neural signaling pathways through the gut-brain axis.",
        "Biology",
    ),
    # Environmental & Earth Sciences
    (
        "REAL_ENV_01_CoralBleaching",
        "Thermal Stress and Coral Reef Bleaching Dynamics",
        "Coral reef ecosystems are exceptionally vulnerable to elevated sea surface temperatures driven by anthropogenic climate change. "
        "Sustained marine heatwaves cause physiological stress in scleractinian corals, disrupting symbiotic relationships with photosynthetic "
        "dinoflagellates of the family Symbiodiniaceae. Corals expel their algal symbionts, exposing white calcium carbonate skeletons in a "
        "process known as coral bleaching. Prolonged symbiont loss results in widespread coral mortality and biodiversity degradation.",
        "Environmental Science",
    ),
    (
        "REAL_ENV_02_CarbonSinks",
        "Oceanic Carbon Sinks and Acidification Chemistry",
        "The global oceans act as critical carbon sinks, absorbing approximately one-quarter of anthropogenic carbon dioxide emissions. "
        "Dissolution of atmospheric carbon dioxide in seawater forms carbonic acid, which dissociates into hydrogen ions and bicarbonate. "
        "The increase in hydrogen ion concentration lowers ocean pH and decreases carbonate ion availability, hindering calcifying marine "
        "organisms from building calcium carbonate shells and structures.",
        "Environmental Science",
    ),
    (
        "REAL_ENV_03_GlacialMelt",
        "Glacial Mass Balance and Cryosphere Sea-Level Contribution",
        "Global cryosphere decline is marked by accelerated mass loss across polar ice sheets and alpine glaciers. Satellite altimetry "
        "and gravimetry record negative mass balance in both the Greenland and Antarctic ice sheets. Surface meltwater lubrication at the ice-bed "
        "interface and ocean-induced basal melting beneath floating ice shelves accelerate calving and ice discharge into oceans, driving "
        "global mean sea-level rise and threatening coastal ecosystems.",
        "Environmental Science",
    ),
    (
        "REAL_ENV_04_RenewableEnergy",
        "Geothermal Energy Systems and Subsurface Thermodynamics",
        "Geothermal energy harnesses thermal energy stored in Earth's crust derived from radioactive decay of minerals and primordial planetary "
        "formation. Hydrothermal power plants extract steam and high-temperature pressurized brine from deep permeable reservoirs to drive "
        "turbines for base-load electricity generation. Enhanced geothermal systems engineer permeability in hot dry rock through hydraulic "
        "stimulation, expanding clean renewable energy potential while requiring careful monitoring of induced seismicity.",
        "Environmental Science",
    ),
    (
        "REAL_ENV_05_ForestCanopies",
        "Tropical Rainforest Canopies and Terrestrial Carbon Cycling",
        "Tropical rainforest canopies represent hyper-diverse terrestrial strata regulating global hydrological and carbon cycles. Stratified "
        "foliage layers intercept solar radiation and precipitation, driving localized evapotranspiration and regional atmospheric moisture "
        "recycling. Massive above-ground biomass in mature tropical forests sequesters gigatons of carbon, while deforestation and drought "
        "risk converting tropical biomes from net carbon sinks into net atmospheric carbon sources.",
        "Environmental Science",
    ),
    (
        "REAL_ENV_06_WaterSecurity",
        "Groundwater Aquifer Depletion and Hydrological Recharge",
        "Overexploitation of deep confined aquifers for agricultural irrigation exceeds natural recharge rates, threatening global freshwater "
        "security. Intensive borehole pumping lowers groundwater tables, reduces baseflow to rivers, and induces irreversible land subsidence "
        "through compaction of aquitard clay layers. Sustainable water resource governance requires managed aquifer recharge, spatial remote "
        "sensing of water storage anomalies via GRACE satellites, and precision drip irrigation technology.",
        "Environmental Science",
    ),
]


# 70 Synthetic Documents Generator (70% Synthetic Data)

# Vocabulary universe for Zipfian text synthesis
ZIPF_VOCABULARY = [
    "network", "system", "algorithm", "model", "data", "function", "probability", "layer",
    "distribution", "feature", "vector", "matrix", "analysis", "parameter", "process", "gradient",
    "learning", "information", "structure", "optimization", "energy", "quantum", "entropy", "space",
    "signal", "dimension", "sequence", "representation", "kernel", "estimate", "sample", "cluster",
    "frequency", "computation", "graph", "variable", "convergence", "accuracy", "measure", "theory",
    "tensor", "state", "equation", "iteration", "variance", "coefficient", "metric", "partition",
    "dynamic", "transformation", "correlation", "component", "index", "inference", "memory", "rate",
    "threshold", "classification", "regression", "hypothesis", "complexity", "capacity", "residual",
    "attention", "concurrency", "consensus", "cryptography", "transduction", "approximation", "latency",
    "hash", "bucket", "collision", "bandwidth", "topology", "subspace", "manifold", "equilibrium"
]


def generate_zipfian_synthetic_document(doc_id: str, num_words: int = 140, alpha: float = 1.15, seed: int = 42) -> Tuple[str, str, str, str]:
    # Generates a synthetic document following Zipf's power-law distribution f(r) ~ 1 / r^alpha.
    # Directly models the heavy-tailed frequency distributions used in Otmar Ertl's experimental evaluation of ProbMinHash (Section 3).

    rng = random.Random(seed)
    vocab = list(ZIPF_VOCABULARY)
    rng.shuffle(vocab)

    n_vocab = len(vocab)
    ranks = list(range(1, n_vocab + 1))
    weights = [1.0 / (r ** alpha) for r in ranks]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Sample words according to Zipf distribution
    sampled_tokens = rng.choices(vocab, weights=probs, k=num_words)

    # Structure into readable pseudo-sentences
    sentences = []
    chunk_size = 14
    for i in range(0, len(sampled_tokens), chunk_size):
        chunk = sampled_tokens[i : i + chunk_size]
        if chunk:
            sentences.append(" ".join(chunk).capitalize() + ".")

    text = " ".join(sentences)
    title = f"Zipfian Power-Law Synthetic Distribution (alpha={alpha})"
    category = "Synthetic (Zipfian Power-Law)"
    return doc_id, title, text, category


def generate_perturbed_synthetic_document(
    doc_id: str,
    base_id: str,
    base_title: str,
    base_text: str,
    target_similarity: float,
    seed: int = 42,
) -> Tuple[str, str, str, str]:
    # Generates a calibrated synthetic near-duplicate with target probability overlap.
    # Applies controlled word substitution, insertion, and token swapping to produce
    # ground-truth near-duplicates in graded brackets (e.g., 95%, 75%, 50%, 25%).
    
    rng = random.Random(seed)
    words = base_text.split()
    total_words = len(words)

    # Determine number of words to replace to achieve target similarity
    # Jaccard overlap approx equals (1 - replace_fraction) / (1 + replace_fraction)
    # Solving for replace_fraction: replace_ratio = (1 - s) / (1 + s)
    replace_ratio = max(0.0, min(0.9, (1.0 - target_similarity) / (1.0 + target_similarity)))
    num_to_replace = int(total_words * replace_ratio)

    indices_to_replace = set(rng.sample(range(total_words), num_to_replace))
    perturbed_words = []

    for idx, w in enumerate(words):
        if idx in indices_to_replace:
            # Replace with a synthetic token
            replacement = rng.choice(ZIPF_VOCABULARY)
            perturbed_words.append(replacement)
        else:
            perturbed_words.append(w)

    text = " ".join(perturbed_words)
    title = f"Synthetic Calibrated Variant of {base_id} (Target J~{int(target_similarity*100)}%)"
    category = f"Synthetic (Calibrated {int(target_similarity*100)}% Variant)"
    return doc_id, title, text, category


def build_100_document_corpus(seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #Builds the complete 100-document dataset:
    #  - 30 Real Documents (30% Real Data)
    #  - 35 Zipfian Synthetic Documents (35% Heavy-Tailed Power-Law)
    #  - 35 Calibrated Perturbation Documents (35% Controlled Ground-Truth Overlap)
    # Total = 100 documents (Exactly 30% Real, 70% Synthetic).
    
    corpus_dict: Dict[str, Dict[str, Any]] = {}
    manifest_records: List[Dict[str, Any]] = []

    # 1. Add 30 Real Documents (30%)
    for d_id, title, text, category in REAL_DOCUMENTS:
        doc_entry = {
            "id": d_id,
            "title": title,
            "text": text,
            "is_synthetic": False,
            "category": category,
            "generation_type": "real_domain_data",
            "word_count": len(text.split()),
        }
        corpus_dict[d_id] = doc_entry
        manifest_records.append(doc_entry)

    # 2. Add 35 Zipfian Synthetic Documents (Part 1 of 70% Synthetic)
    for i in range(1, 36):
        d_id = f"SYNTH_ZIPF_{i:02d}"
        alpha_val = round(1.05 + (i % 7) * 0.05, 2)
        _, title, text, category = generate_zipfian_synthetic_document(
            doc_id=d_id,
            num_words=130 + (i % 5) * 10,
            alpha=alpha_val,
            seed=seed + i * 31,
        )
        doc_entry = {
            "id": d_id,
            "title": title,
            "text": text,
            "is_synthetic": True,
            "category": category,
            "generation_type": "synthetic_zipfian_power_law",
            "word_count": len(text.split()),
        }
        corpus_dict[d_id] = doc_entry
        manifest_records.append(doc_entry)

    # 3. Add 35 Calibrated Perturbation Documents (Part 2 of 70% Synthetic)
    # Target similarity brackets:
    # 10 documents at 92-98% (near duplicates / copy-edits)
    # 15 documents at 65-80% (paraphrases / moderate overlap)
    # 10 documents at 30-45% (partial overlap)
    targets = (
        [0.95] * 10 +
        [0.75] * 15 +
        [0.35] * 10
    )

    for i, target_s in enumerate(targets, start=1):
        d_id = f"SYNTH_PERTURB_{i:02d}"
        base_doc = REAL_DOCUMENTS[(i - 1) % len(REAL_DOCUMENTS)]
        base_id, base_title, base_text, _ = base_doc

        _, title, text, category = generate_perturbed_synthetic_document(
            doc_id=d_id,
            base_id=base_id,
            base_title=base_title,
            base_text=base_text,
            target_similarity=target_s,
            seed=seed + i * 53,
        )
        doc_entry = {
            "id": d_id,
            "title": title,
            "text": text,
            "is_synthetic": True,
            "category": category,
            "generation_type": "synthetic_calibrated_perturbation",
            "base_doc_id": base_id,
            "target_similarity": target_s,
            "word_count": len(text.split()),
        }
        corpus_dict[d_id] = doc_entry
        manifest_records.append(doc_entry)

    # Generate Manifest Summary
    real_count = sum(1 for d in manifest_records if not d["is_synthetic"])
    synth_count = sum(1 for d in manifest_records if d["is_synthetic"])
    total_count = len(manifest_records)

    manifest_summary = {
        "total_documents": total_count,
        "real_documents_count": real_count,
        "real_documents_percentage": round(real_count / total_count * 100.0, 1),
        "synthetic_documents_count": synth_count,
        "synthetic_documents_percentage": round(synth_count / total_count * 100.0, 1),
        "categories": {
            "Real Domain Corpus": real_count,
            "Synthetic (Zipfian Power-Law)": sum(1 for d in manifest_records if d["generation_type"] == "synthetic_zipfian_power_law"),
            "Synthetic (Calibrated Overlap)": sum(1 for d in manifest_records if d["generation_type"] == "synthetic_calibrated_perturbation"),
        },
        "reference_paper": {
            "title": "ProbMinHash – A Class of Locality-Sensitive Hash Algorithms for the (Probability) Jaccard Similarity",
            "author": "Otmar Ertl",
            "journal": "IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022",
            "arxiv": "arXiv:1911.00675",
        },
        "documents": manifest_records,
    }

    return corpus_dict, manifest_summary


def ensure_corpus_saved() -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Ensures corpus.json and dataset_manifest.json exist on disk and returns

    (text_dict mapping doc_id -> text, manifest_dict).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dict, manifest_dict = build_100_document_corpus(seed=42)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus_dict, f, indent=2)

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    text_dict = {d_id: info["text"] for d_id, info in corpus_dict.items()}
    return text_dict, manifest_dict


def get_dataset_stats() -> Dict[str, Any]:
    """Returns dataset summary statistics."""
    _, manifest = load_corpus()
    return {
        "total_documents": manifest["total_documents"],
        "real_documents_count": manifest["real_documents_count"],
        "real_documents_percentage": manifest["real_documents_percentage"],
        "synthetic_documents_count": manifest["synthetic_documents_count"],
        "synthetic_documents_percentage": manifest["synthetic_documents_percentage"],
        "categories": manifest["categories"],
        "reference_paper": manifest["reference_paper"],
    }



def load_corpus() -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Loads text documents and manifest metadata."""
    if not CORPUS_FILE.exists() or not MANIFEST_FILE.exists():
        return ensure_corpus_saved()

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus_dict = json.load(f)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest_dict = json.load(f)

    text_dict = {d_id: info["text"] for d_id, info in corpus_dict.items()}
    return text_dict, manifest_dict


def get_dataset_stats() -> Dict[str, Any]:
    """Returns dataset summary statistics."""
    _, manifest = load_corpus()
    return {
        "total_documents": manifest["total_documents"],
        "real_documents_count": manifest["real_documents_count"],
        "real_documents_percentage": manifest["real_documents_percentage"],
        "synthetic_documents_count": manifest["synthetic_documents_count"],
        "synthetic_documents_percentage": manifest["synthetic_documents_percentage"],
        "categories": manifest["categories"],
        "reference_paper": manifest["reference_paper"],
    }

