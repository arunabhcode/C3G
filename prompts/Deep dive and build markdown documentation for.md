# Deep dive and build markdown documentation for ...

Exported: 6/8/2026, 12:07:49 AM

---

**User:**

Create the tasks for the repo-documentation

---

**Assistant:**

### Sub-Agent: Feature Requirements First Workflow

**Explanation:** Delegating to the requirements-first workflow subagent to gather and document requirements for creating comprehensive repo documentation.

**Prompt:**

The user wants to create comprehensive markdown documentation for every part of the C3G repository so they can completely understand the codebase, including what to read, sources that explain what's going on in the code, etc.

Spec type: New Feature
Feature name: repo-documentation

Context about the repository:
- C3G is a research/ML codebase (appears to be related to 3D reconstruction, Gaussian splatting, and novel view synthesis based on the folder structure - config/model/encoder/noposplat, decoder/splatting_cuda, datasets like re10k, scannet, dl3dv, dtu)
- Located at /home/arunabh/C3G
- Has a README.md, LICENSE, pyproject.toml, requirements.txt
- Main source code in src/ directory with submodules: dataset, evaluation, geometry, loss, misc, and more
- Configuration managed via YAML files in config/ (Hydra-based)
- Training configs suggest multiple feature heads (DINOv2 B/L, DINOv3 L, LSEG, VGGT) and gaussian heads
- Assets include evaluation indices for ACID, DTU, RE10K, ScanNet++

The user's goal is a "deep dive" — comprehensive documentation that covers every part of the repo, helps them fully understand:
1. What the repo does (purpose, research contribution)
2. Architecture and module organization
3. What files/docs to read in what order
4. External sources (papers, references) that explain what's going on
5. How to run/train/evaluate
6. Key abstractions and code flow

The output feature will be a set of markdown documentation files (likely under a docs/ directory) covering: overview, getting started, architecture, modules, configuration system, datasets, models, training, evaluation, and a reading guide with external references.

Please gather requirements by reviewing the repo structure, README, key source files (src/main.py, src/config.py, src/model/*, src/dataset/*), and the configuration files to understand what needs to be documented. Then produce the requirements document.

**Response:**

Completed Phase 1 (Requirements) of the requirements-first workflow for the `repo-documentation` feature.

Artifacts created:
- `/home/arunabh/C3G/.kiro/specs/repo-documentation/.config.kiro` (workflow config)
- `/home/arunabh/C3G/.kiro/specs/repo-documentation/requirements.md` (15 requirements, EARS-compliant, INCOSE-vetted)

Scope: Produce a `docs/` tree of Markdown that lets a new reader fully understand C3G (CVPR 2026, arXiv 2512.04021). Requirements cover: overview, doc index, getting-started with Run_Recipes, architecture with data-flow diagram, a Module_Document for every source package (including backbone/croco, backbone/vggt, heads, common, visualization, decoder, distiller, clip, transformer, encodings, loss, evaluation, geometry, misc, visualization, scripts), Hydra config-system doc, datasets & view samplers, models & foundation models, training/losses/optimizer, evaluation/metrics/pose-alignment, a reading guide with interleaved External_References, code-flow walkthroughs for train/test/feature-head runs, glossary, fidelity-to-code constraints, and strict file-layout rules under `docs/`.

Ran `analyze_requirements` and applied all three improvements:
- AC 3.6 tightened to require exact `NUM_SEMANTIC_CHANNELS` values.
- AC 13.3 broadened to require references for every glossary term.
- AC 15.2 and 15.3 rewritten with `WHERE` clauses so format rules are conditional on the correct `docs/` location.

Ready for user review before moving to Phase 2 (Design).

Now I need to run the automatic requirements detailing step. Let me invoke the requirement-detailer for each requirement in parallel.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 1 to make it more precise and testable

**Prompt:**

### Requirement 1: Repository Overview Document

**User Story:** As a new reader of C3G, I want a single starting document that explains what the project does and how to begin, so that I can orient myself before touching any code.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/01-overview.md`.
2. THE `docs/01-overview.md` file SHALL describe the research contribution of C3G in prose, including the claim that the model predicts 2K Gaussians from unposed images.
3. THE `docs/01-overview.md` file SHALL include External_References to the C3G arXiv paper (2512.04021), the C3G project page, and the two acknowledged upstream projects VGGT and NoPoSplat.
4. THE `docs/01-overview.md` file SHALL list the top-level directories of the C3G_Repository (`src/`, `config/`, `assets/`, `docs/`, `pretrained_weights/`) with a one-line description of each.
5. THE `docs/01-overview.md` file SHALL state which Python version, PyTorch version, and CUDA version the code targets, as declared in `README.md` and `requirements.txt`.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 1 in-place. Key changes: criterion 2 now enumerates four required content claims (feed-forward, unposed input, ~2,000 Gaussians, NVS + 3D scene understanding via feature distillation) instead of leaving "describe ... in prose" open-ended; criterion 3 pins each External_Reference to a concrete URL derived from README.md so testers can verify link targets; criterion 4 adds a 120-character bound per entry and makes each directory a separate entry; criterion 5 names the exact version values (Python 3.11, PyTorch 2.5.1, CUDA 12.4) from README.md; added one new IF-THEN criterion (criterion 6) to handle the case where README.md and requirements.txt disagree with the pinned values, keeping the refinement within the 3-new-criteria budget.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 2 to make it more precise and testable

**Prompt:**

### Requirement 2: Documentation Index

**User Story:** As a reader browsing the `docs/` folder, I want a table of contents, so that I can find the right document without opening each file.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a Documentation_Index at `docs/README.md`.
2. THE Documentation_Index SHALL list every other file in the Documentation_Set with a relative Markdown link and a one-sentence summary.
3. THE Documentation_Index SHALL present documents in a numbered order that matches the recommended reading sequence defined in the Reading_Guide.
4. WHERE a new document is added to the Documentation_Set, THE Documentation_Index SHALL include a corresponding entry for that document.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 2 (Documentation Index). Amended criteria 2, 3, and 4 to eliminate vagueness and fix an EARS pattern misuse (WHERE → WHEN for an event trigger), and added one new criterion covering the `docs/modules/` section layout.

Key changes:
- Criterion 2: Bounded "one-sentence summary" to 10–200 characters and explicitly scoped the listing to every `.md` file under `docs/` and `docs/modules/`, excluding `docs/README.md` itself.
- Criterion 3: Specified the numbered-list format and required the numeric order to match the two-digit `NN-` filename prefix plus the Reading_Guide sequence.
- New Criterion 4: Requires a dedicated, alphabetically ordered section for Module_Documents under `docs/modules/` with bounded summaries and relative links.
- Criterion 5 (replaces original 4): Switched EARS pattern from WHERE (optional feature) to WHEN (event-driven for "new document added") and specified the exact entry format with the same length bounds.

Net addition: 1 new criterion (within the max-3 budget). All P0/P1 gaps from the vagueness/gap analysis are addressed; no implementation details introduced.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 3 to make it more precise and testable

**Prompt:**

### Requirement 3: Getting Started and Run Recipes

**User Story:** As a user who wants to run C3G, I want exact commands for installation, data preparation, training, and evaluation, so that I can reproduce results without guessing.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/02-getting-started.md`.
2. THE `docs/02-getting-started.md` file SHALL reproduce the conda and pip installation commands from `README.md` verbatim inside fenced code blocks.
3. THE `docs/02-getting-started.md` file SHALL document the download steps for the VGGT pretrained weight (`pretrained_weights/model.pt`) and the LSeg pretrained weight (`pretrained_weights/demo_e200.ckpt`).
4. THE `docs/02-getting-started.md` file SHALL list every pretrained checkpoint published on the C3G Hugging Face repository (`gaussian_decoder.ckpt`, `gaussian_decoder_multiview.ckpt`, `feature_decoder_lseg.ckpt`, `feature_decoder_dinov3L.ckpt`, `feature_decoder_dinov2.ckpt`) with a one-line description of each.
5. THE `docs/02-getting-started.md` file SHALL include a Run_Recipe for each of the following workflows, each inside its own fenced code block: gaussian-head training (2-view), gaussian-head training (multi-view), gaussian-head multi-view continued from a 2-view checkpoint, feature-head training for LSeg, DINOv2-base, DINOv2-large, DINOv3-large, VGGT-tracking, feature-head multi-view training for LSeg, RE10K 2-view evaluation, RE10K multi-view evaluation, and ScanNet 3D-scene-understanding evaluation.
6. THE `docs/02-getting-started.md` file SHALL document the `NUM_SEMANTIC_CHANNELS` CUDA rasterizer configuration requirement, using these exact per-model values as declared in `README.md`: 512 for LSeg, 768 for DINOv2-base, 1024 for DINOv2-large and DINOv3-large, and 128 for VGGT-tracking.
7. IF a Run_Recipe requires an environment variable, a pretrained weight, or a dataset path, THEN THE `docs/02-getting-started.md` file SHALL state that requirement adjacent to the recipe.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 3 with tighter wording on download steps (Criterion 3), fidelity on Run_Recipes (Criterion 5), precise adjacency for recipe-requirement prose (Criterion 7), and two new criteria addressing the P0 data-preparation gap and section ordering gap.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 4 to make it more precise and testable

**Prompt:**

### Requirement 4: Architecture Overview

**User Story:** As a reader who wants to understand C3G end-to-end, I want a diagram and narrative that trace a batch from dataset through encoder, decoder, and loss, so that I can form a mental model of the system before diving into individual files.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/03-architecture.md`.
2. THE `docs/03-architecture.md` file SHALL describe the forward path from context images to rendered target images, naming each Public_Symbol that participates: `DataModule`, `DatasetRE10k` (and peers), `ModelWrapper`, `EncoderVGGT` or `EncoderNoPoSplat`, `UnifiedGaussianAdapter`, `DecoderSplattingCUDA.render_cuda`, and `Loss` subclasses.
3. THE `docs/03-architecture.md` file SHALL describe the data type flow using `BatchedExample` and `Gaussians` dataclasses as defined in `src/dataset/types.py` and `src/model/types.py`.
4. THE `docs/03-architecture.md` file SHALL contain a Mermaid or ASCII diagram that shows the modules in Acceptance Criterion 2 connected by arrows labelled with the data type passed between them.
5. THE `docs/03-architecture.md` file SHALL describe how the Foundation_Model features are injected during feature-head training, specifically referencing `ModelWrapper.forward_foundation_model` and the `InstillTransformer` path in `src/model/encoder/common/gmae.py`.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 4 (Architecture Overview). Amended criteria 2–5 to remove vagueness ("and peers", "Loss subclasses", unspecified diagram labels, unspecified dataclass detail, unspecified injection depth), enumerated the peer Dataset_Adapters (`DatasetScannet`, `DatasetScannetPose`, `DatasetReplica`) and Loss subclasses (`LossMse`, `LossLpips`) grounded in the actual codebase, specified the minimum diagram labels (`BatchedExample`, `Gaussians`), and added one new criterion (6) to cover the encoder-selection rule gap. No implementation details beyond what already existed in the original requirement were added.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 5 to make it more precise and testable

**Prompt:**

### Requirement 5: Source Tree Module Documents

**User Story:** As a contributor editing a specific subsystem, I want one document per top-level source package that describes every file in that package, so that I can locate the right file to modify.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain one Module_Document for each of the following packages: `src/` (root), `src/dataset/`, `src/dataset/view_sampler/`, `src/dataset/shims/`, `src/model/`, `src/model/encoder/`, `src/model/encoder/backbone/`, `src/model/encoder/backbone/croco/`, `src/model/encoder/backbone/vggt/`, `src/model/encoder/heads/`, `src/model/encoder/common/`, `src/model/encoder/visualization/`, `src/model/decoder/`, `src/model/distiller/`, `src/model/clip/`, `src/model/transformer/`, `src/model/encodings/`, `src/loss/`, `src/evaluation/`, `src/geometry/`, `src/misc/`, `src/visualization/`, `src/visualization/camera_trajectory/`, `src/visualization/drawing/`, and `src/scripts/`.
2. FOR each Module_Document, THE document SHALL list every `.py` file in its corresponding package with a one-to-three-sentence description of the file's purpose.
3. FOR each Module_Document, THE document SHALL identify the Public_Symbols exported by the package's `__init__.py` when such a file exists.
4. FOR each Module_Document, THE document SHALL include at least one External_Reference when the package is derived from or closely based on an external codebase (for example CroCo, VGGT, DUSt3R, NoPoSplat, pixelSplat, MVSplat, LSeg, CLIP, gsplat, Feature 3DGS).
5. WHERE a source file contains a `README.md` authored by an upstream project (for example `src/model/clip/README.md`, `src/model/encoder/backbone/croco/README.md`), THE corresponding Module_Document SHALL link to that upstream README.
6. THE Module_Documents SHALL be placed under `docs/modules/` and named to match the package path (for example `docs/modules/model-encoder.md` for `src/model/encoder/`).

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 5 to eliminate EARS pattern misuse (converted "FOR each ... when" constructs to proper ubiquitous and WHERE patterns), clarified the vague verb "identify" in criterion 3 with an explicit rule for what counts as an exported Public_Symbol (`__all__` or top-level imports/definitions), made criterion 2 explicit about including `__init__.py`, tightened criterion 5's "authored by an upstream project" wording and required a repository-relative Markdown link, and replaced the example-only naming rule in criterion 6 with a deterministic path-to-filename transformation (relative path with `/` replaced by `-`, `.md` appended) so two testers would produce identical pass/fail results. No new criteria were added; all six original criteria were amended in place.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 6 to make it more precise and testable

**Prompt:**

### Requirement 6: Configuration System Document

**User Story:** As a user who wants to tweak experiments, I want a document that explains the Hydra configuration tree, so that I know where to change hyperparameters and how configs compose.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a Config_System_Document at `docs/04-configuration.md`.
2. THE `docs/04-configuration.md` file SHALL describe the role of `config/main.yaml` as the Hydra entry configuration loaded by `src/main.py`.
3. THE `docs/04-configuration.md` file SHALL list every file under `config/` organized by subdirectory (`dataset/`, `dataset/view_sampler/`, `dataset/view_sampler_dataset_specific_config/`, `evaluation/`, `loss/`, `model/encoder/`, `model/encoder/backbone/`, `model/decoder/`, `training/`) with a one-line description of what each file configures.
4. THE `docs/04-configuration.md` file SHALL document the Hydra `defaults` composition used by each file in `config/training/` and `config/evaluation/`, including which dataset and which encoder/decoder overrides each one applies.
5. THE `docs/04-configuration.md` file SHALL map each top-level config field consumed by `src/config.py` (`wandb`, `mode`, `dataset`, `data_loader`, `model`, `optimizer`, `checkpointing`, `trainer`, `loss`, `test`, `train`, `seed`) to the Python dataclass that parses it.
6. THE `docs/04-configuration.md` file SHALL describe how `cfg.train.reproj_model` selects the Foundation_Model loaded by `src/model/load_foundation_model.py` and lists the accepted string values (`vggt`, `vggt_tracking`, `dinov2_B`, `dinov2_L`, `dinov3_L`, `dinov3_H`, `dinov3_7B`, `lseg`, `maskclip`).

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 6 (Configuration System Document) by addressing the following gaps identified via EARS vagueness/pattern scan and backward reasoning:

- Criterion 2: "describe the role" was vague — tightened to require identifying `hydra.main` loading and reproducing the top-level `defaults` entries.
- Criterion 3: "list every file" — tightened to require repository-relative paths, grouping by subdirectory, and one-sentence descriptions; also clarified that `config/main.yaml` is included.
- Criterion 4: "document the Hydra `defaults` composition" — tightened to require reproducing the `defaults` list verbatim and naming the dataset/encoder/decoder/loss overrides explicitly.
- Criterion 5: map-to-dataclass — specified as a two-column table with fully-qualified dataclass names, making verification unambiguous.
- Criterion 6: "describe how" — specified as a table listing each accepted `reproj_model` string and the Foundation_Model it resolves to.
- Added one new criterion (7) to handle the error case where a listed YAML file no longer exists, keeping the doc consistent with the repo (supports Req 14 fidelity).

Budget compliance: 1 new criterion (≤ 3), all vague qualifiers eliminated, no implementation details added, all criteria follow EARS patterns.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 7 to make it more precise and testable

**Prompt:**

### Requirement 7: Datasets and View Sampling Document

**User Story:** As a user preparing data for C3G, I want a document that describes each supported dataset and how view pairs are sampled, so that I can prepare data and choose the correct view sampler for my experiment.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/05-datasets.md`.
2. THE `docs/05-datasets.md` file SHALL describe each Dataset_Adapter (`DatasetRE10k`, `DatasetScannet`, `DatasetScannetPose`, `DatasetReplica`) including the dataset name, the file it lives in, the expected chunk format, and the default near/far planes where declared in code.
3. THE `docs/05-datasets.md` file SHALL list the four view samplers (`all`, `arbitrary`, `bounded`, `evaluation`) with a one-paragraph description of each that matches its implementation in `src/dataset/view_sampler/`.
4. THE `docs/05-datasets.md` file SHALL list the four evaluation-index JSON files under `assets/` (`evaluation_index_acid.json`, `evaluation_index_dtu.json`, `evaluation_index_re10k.json`, `evaluation_index_scannetpp.json`) with External_References to the originating dataset homepages.
5. THE `docs/05-datasets.md` file SHALL include External_References to the preprocessing sources named in `README.md`: pixelSplat, MVSplat, LSM (for ScanNet), and Feature 3DGS (for Replica).
6. THE `docs/05-datasets.md` file SHALL describe each shim in `src/dataset/shims/` (`augmentation_shim`, `bounds_shim`, `crop_shim`, `normalize_shim`, `patch_shim`) in one sentence each.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 7 to eliminate vagueness and close critical gaps:

- Criterion 2: split the "describe each Dataset_Adapter" sentence into a precise list (display name, file path, expected top-level chunk keys read by `__getitem__`, explicit near/far from config dataclass or class constants). Replaces vague "expected chunk format" and "where declared in code".
- Added Criterion 3 (new, P0): handles the fallback when an adapter does not declare its own near/far, requiring the doc to state inheritance and cite the base symbol. Closes a test-blocker gap from criterion 2's "where declared in code" clause.
- Criterion 4 (was 3): bounded "one-paragraph description" to 40–120 words and made required content explicit (source file, context/target view counts, selection rule). Eliminates the vague qualifier "matches its implementation".
- Criterion 5 (was 4): tightened by explicitly pairing each JSON file with its originating dataset homepage (ACID, DTU, RealEstate10K, ScanNet++).
- Criterion 7 (was 6): replaced "one sentence each" with "a single sentence of 15 to 40 words" and required content (transformation applied + field(s) modified on `BatchedExample`).

Budget: 1 new criterion added (well under the 3-criterion cap), all others are amendments. No implementation details or new technologies introduced; preserved existing symbol names and file paths from the original.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 8 to make it more precise and testable

**Prompt:**

### Requirement 8: Models and Foundation Models Document

**User Story:** As a researcher extending the model, I want a document that explains the encoder, decoder, Gaussian representation, and Foundation_Model integration, so that I can swap components without reading every file.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/06-models.md`.
2. THE `docs/06-models.md` file SHALL describe the three encoder variants registered in `src/model/encoder/__init__.py` (`noposplat`, `noposplat_multi`, `vggt`) and the single decoder variant (`splatting_cuda`).
3. THE `docs/06-models.md` file SHALL describe the `Gaussians` dataclass in `src/model/types.py`, naming each field and its tensor shape as declared with `jaxtyping`.
4. THE `docs/06-models.md` file SHALL describe the `UnifiedGaussianAdapter` and `GaussianAdapter` classes in `src/model/encoder/common/gaussian_adapter.py` and the spherical-harmonics configuration controlled by `gaussian_adapter.sh_degree`.
5. THE `docs/06-models.md` file SHALL describe the four DPT-style heads in `src/model/encoder/heads/` (`dpt_head`, `dpt_gs_head`, `vggt_dpt_head`, `vggt_dpt_gs_head`) and the `linear_head` variant.
6. THE `docs/06-models.md` file SHALL describe the `Transformer` and `InstillTransformer` classes in `src/model/encoder/common/gmae.py` and their role in producing the 2K Gaussian tokens.
7. THE `docs/06-models.md` file SHALL document `src/model/load_foundation_model.py`, listing each supported Foundation_Model, its Hugging Face or PyTorch-Hub identifier as declared in code, and its output `feature_dim`.
8. THE `docs/06-models.md` file SHALL include External_References to the VGGT, DUSt3R, CroCo, NoPoSplat, DINOv2, DINOv3, LSeg, CLIP, MaskCLIP, and Feature 3DGS source papers or repositories.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 8 acceptance criteria to eliminate vague "describe" verbs and specify testable content for each criterion. Key changes: added explicit content requirements (class name, file path, summary length) to criteria 2/4/5/6, pinned jaxtyping shape format in criterion 3, clarified sh_degree accepted values in criterion 4, required citation of the "2K" figure to source in criterion 6, specified link format for External_References in criterion 8, and added criterion 9 as a fidelity guard against fabricated references.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 9 to make it more precise and testable

**Prompt:**

### Requirement 9: Training, Losses, and Optimization Document

**User Story:** As a user running a training job, I want a document that explains the training loop, the available losses, and the optimizer configuration, so that I can understand training logs and modify the loss set.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/07-training.md`.
2. THE `docs/07-training.md` file SHALL describe the entry point `src/main.py` including how it loads configs via `hydra.main`, constructs the Lightning `Trainer`, and dispatches to `trainer.fit` when `cfg.mode == "train"` and `trainer.test` otherwise.
3. THE `docs/07-training.md` file SHALL describe `ModelWrapper.training_step` in `src/model/model_wrapper.py`, including how losses are iterated and how `feature_rendering_loss` interacts with `train.reproj_model`.
4. THE `docs/07-training.md` file SHALL list the losses registered in `src/loss/__init__.py` (`LossMse`, `LossLpips`) and describe each additional loss file present in `src/loss/` (`loss_point.py`, `loss_ss.py`, `loss_ssim.py`) with its purpose.
5. THE `docs/07-training.md` file SHALL describe the `OptimizerCfg` fields (`lr`, `warm_up_steps`, `backbone_lr_multiplier`) and the `TrainerCfg` fields (`max_steps`, `val_check_interval`, `gradient_clip_val`, `num_nodes`, `accumulate_grad_batches`) as they are consumed in `src/main.py`.
6. THE `docs/07-training.md` file SHALL describe the checkpoint-loading logic in `src/main.py` that handles the three checkpoint formats (`model`, `state_dict`, and the VGGT point-head/aggregator format).

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 9 to eliminate vague uses of "describe" by quantifying required content per criterion and added one new criterion (P0) covering the unhandled-checkpoint-format error path. Amendments specify verbatim identifiers, enumerated content items, and length bounds (one to three sentences) so two testers reach identical pass/fail results. Budget: 1 new criterion added (under the 3-criterion cap); all other gaps handled via amendments.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 10 to make it more precise and testable

**Prompt:**

### Requirement 10: Evaluation and Metrics Document

**User Story:** As a user evaluating a trained model, I want a document that explains each evaluation mode and metric, so that I can interpret the outputs of `src/main.py` in test mode and `src/eval_pose.py`.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/08-evaluation.md`.
2. THE `docs/08-evaluation.md` file SHALL describe the `TestCfg` dataclass fields consumed by `ModelWrapper.test_step`, including `align_pose`, `pose_align_steps`, `rot_opt_lr`, `trans_opt_lr`, `compute_scores`, `save_image`, `save_video`, `save_compare`, `forward_vfm`, `labels`, and `color_hex_list`.
3. THE `docs/08-evaluation.md` file SHALL describe the metrics defined in `src/evaluation/metrics.py` (PSNR, SSIM, LPIPS) and how `compute_lpips`, `compute_psnr`, `compute_ssim` are called from `ModelWrapper.test_step`.
4. THE `docs/08-evaluation.md` file SHALL describe the `PoseEvaluator` class in `src/evaluation/pose_evaluator.py` and the separate `src/eval_pose.py` entry point.
5. THE `docs/08-evaluation.md` file SHALL describe the `EvaluationIndexGenerator` in `src/evaluation/evaluation_index_generator.py` and how the generated indices relate to the JSON files under `assets/`.
6. THE `docs/08-evaluation.md` file SHALL describe the ScanNet 3D-scene-understanding evaluation flow, including the mIoU and accuracy computation via `torchmetrics.JaccardIndex` and `torchmetrics.Accuracy` in `ModelWrapper`.
7. THE `docs/08-evaluation.md` file SHALL describe the output directory layout produced by a test run, including the `outputs/exp_<name>/<timestamp>/` path defined by `hydra.run.dir` in `config/main.yaml`, and the per-scene artifacts written by `test_step` (`color/`, `seg/`, `seg_gt/`, `*_projections.png`, comparison images).

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 10 with added specificity for each acceptance criterion. Key refinements:

- Criterion 2: Replaced vague "describe ... fields" with a four-part structured requirement (field name, Python type, default value, behavioral description) so testers can unambiguously verify completeness for each of the 11 fields.
- Criterion 3: Replaced vague "describe the metrics ... and how ... are called" with explicit per-metric requirements (function name, input tensor shapes/ranges, scalar output, citation of the invocation site).
- Criterion 4: Added concrete testable sub-items (constructor arguments, listed public methods, quoted CLI invocation, named config file) for the `PoseEvaluator` and `eval_pose.py` entry point.
- Criterion 5: Added explicit listing of the four `assets/*.json` files and a per-file origin classification (generated vs upstream).
- Criterion 6: Replaced "describe the flow" with concrete artifacts to document (call order of methods, citation of instantiation sites, number of classes, metric labels).
- Criterion 7: Split the directory-layout requirement into three testable parts (quote `hydra.run.dir`, list subpaths, map each subpath to the gating `TestCfg` flag).
- Added Criterion 8 as a global fidelity rule requiring source-file + symbol citations for every code-level claim, eliminating residual vagueness across the whole document.

No new technologies introduced; all references (`ModelWrapper.test_step`, `torchmetrics.JaccardIndex`, `torchmetrics.Accuracy`, `hydra.run.dir`, `config/main.yaml`, `config/evaluation/*.yaml`) are preserved from the original requirement or already present in the broader requirements document. Budget respected: 1 new criterion added (under the 3-criterion cap); remaining changes are amendments to existing criteria.

```markdown
### Requirement 10: Evaluation and Metrics Document

**User Story:** As a user evaluating a trained model, I want a document that explains each evaluation mode and metric, so that I can interpret the outputs of `src/main.py` in test mode and `src/eval_pose.py`.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/08-evaluation.md`.
2. THE `docs/08-evaluation.md` file SHALL document the `TestCfg` dataclass fields consumed by `ModelWrapper.test_step` by providing, for each of the fields `align_pose`, `pose_align_steps`, `rot_opt_lr`, `trans_opt_lr`, `compute_scores`, `save_image`, `save_video`, `save_compare`, `forward_vfm`, `labels`, and `color_hex_list`: (a) the field name, (b) the Python type as declared in the dataclass, (c) the default value if declared in code or in `config/evaluation/*.yaml`, and (d) a one-to-three-sentence description of the field's effect on `test_step` behavior.
3. THE `docs/08-evaluation.md` file SHALL document the metrics defined in `src/evaluation/metrics.py` by providing, for each of PSNR, SSIM, and LPIPS: (a) the exact function name (`compute_psnr`, `compute_ssim`, `compute_lpips`), (b) the expected input tensor shapes and value ranges as declared in code, (c) the scalar output produced, and (d) the location in `ModelWrapper.test_step` where the function is invoked, cited as `src/model/model_wrapper.py::ModelWrapper.test_step`.
4. THE `docs/08-evaluation.md` file SHALL document the `PoseEvaluator` class in `src/evaluation/pose_evaluator.py` by listing its constructor arguments, its public methods with a one-sentence description of each, and the `src/eval_pose.py` entry point by quoting the exact command-line invocation pattern (`python -m src.eval_pose` with any Hydra overrides) and naming the config file the entry point consumes.
5. THE `docs/08-evaluation.md` file SHALL document the `EvaluationIndexGenerator` in `src/evaluation/evaluation_index_generator.py` by naming its public method(s), describing the per-scene index structure it produces in one-to-three sentences, and listing by filename each of the four evaluation-index JSON files under `assets/` (`evaluation_index_acid.json`, `evaluation_index_dtu.json`, `evaluation_index_re10k.json`, `evaluation_index_scannetpp.json`) together with a one-sentence statement of whether the file is produced by the generator or supplied from an upstream source.
6. THE `docs/08-evaluation.md` file SHALL document the ScanNet 3D-scene-understanding evaluation flow by naming, in call order, each `ModelWrapper` method involved, and by identifying the exact `torchmetrics.JaccardIndex` and `torchmetrics.Accuracy` instantiation sites in `src/model/model_wrapper.py` (file path and line-containing-symbol), the number of classes configured, and the label each metric is logged under in `test_step` output.
7. THE `docs/08-evaluation.md` file SHALL document the output directory layout produced by a test run by (a) quoting the `hydra.run.dir` string from `config/main.yaml` that defines `outputs/exp_<name>/<timestamp>/`, (b) listing each per-scene artifact subpath written by `test_step` (`color/`, `seg/`, `seg_gt/`, `*_projections.png`, and comparison images), and (c) stating for each subpath the `TestCfg` flag from Criterion 2 that gates its creation.
8. THE `docs/08-evaluation.md` file SHALL cite, for every code-level claim it makes, the exact source file path (relative to the repository root) and the symbol name (class, function, or dataclass field) from which the claim is taken.
```

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 11 to make it more precise and testable

**Prompt:**

### Requirement 11: Reading Guide with External References

**User Story:** As a reader who wants to deeply understand the codebase, I want an ordered reading guide that mixes internal files with papers and reference repositories, so that I can learn the material in the order the authors intend.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a Reading_Guide at `docs/09-reading-guide.md`.
2. THE Reading_Guide SHALL define at least three reader personas (for example "Run the model", "Understand the architecture", "Extend a module") and give each persona a numbered list of files to read.
3. FOR each persona, THE Reading_Guide SHALL interleave at least one External_Reference after every two internal files in its reading list.
4. THE Reading_Guide SHALL include External_References to: the C3G paper (arXiv 2512.04021), the VGGT repository and paper, the NoPoSplat repository, the DUSt3R paper, the CroCo paper, the 3D Gaussian Splatting paper (Kerbl et al.), the pixelSplat and MVSplat papers, the LSeg paper, the DINOv2 and DINOv3 papers, and the CLIP paper.
5. THE Reading_Guide SHALL contain a table mapping each C3G module to the most relevant External_Reference for that module.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 11 to eliminate vagueness and specify testable bounds. Key amendments: (1) added upper bound of 6 personas and minimum of 6 internal files per persona in criterion 2 along with a definition of "internal file"; (2) reframed criterion 3 from non-EARS "FOR each" to proper WHERE pattern and specified the exact interleaving invariant; (3) required External_References in criterion 4 to be Markdown hyperlinks with title plus absolute URL, and split "VGGT repository and paper", "pixelSplat and MVSplat papers", and "DINOv2 and DINOv3 papers" into separate atomic entries; (4) replaced subjective "most relevant" in criterion 5 with an objective scope (rows = Module_Documents from Requirement 5, entries = External_References from criterion 4 or already cited in the module) and required a one-sentence justification per row. No new criteria were added, staying within budget and avoiding any new implementation details.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 12 to make it more precise and testable

**Prompt:**

### Requirement 12: Code Flow Walkthroughs

**User Story:** As a reader who learns best from concrete traces, I want step-by-step walkthroughs of representative runs, so that I can see exactly which functions are called and in what order.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/10-walkthroughs.md`.
2. THE `docs/10-walkthroughs.md` file SHALL contain a walkthrough of a single training step for the `gaussian_head` config, naming each function call in order from `src/main.py:train` through `ModelWrapper.training_step` to `DecoderSplattingCUDA.forward` and the loss computation.
3. THE `docs/10-walkthroughs.md` file SHALL contain a walkthrough of a single test step for the `re10k` evaluation config, naming each function call including `ModelWrapper.test_step_align` when `test.align_pose` is true.
4. THE `docs/10-walkthroughs.md` file SHALL contain a walkthrough of a single feature-head training step for the `feature_head_lseg` config, including the Foundation_Model forward pass through `ModelWrapper.forward_foundation_model` and the feature-rendering-loss branch.
5. EACH walkthrough SHALL cite the exact file path and function name (for example `src/model/model_wrapper.py::ModelWrapper.training_step`) for each step.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 12 (Code Flow Walkthroughs) with the following changes:

- Criterion 2 amended to specify format (ordered numbered list), a minimum step count (≥ 8), explicit start/end points, and the required named calls in order. This removes vagueness about what "walkthrough" means and makes pass/fail testable.
- Criterion 3 amended similarly with ordered list, ≥ 8 steps, start/end points, and the required metric calls named in order.
- New Criterion 4 added to fix the EARS pattern misuse in the original criterion 3 (buried "when test.align_pose is true"): now an Optional-feature (WHERE) criterion that also clarifies behavior when the flag is false.
- Criterion 5 (was original 4) amended with format, ≥ 10 steps, start/end points, and in-order call requirements covering the Foundation_Model path and the feature-rendering-loss branch.
- Criterion 6 (was original 5) rewritten from the non-standard "EACH walkthrough SHALL..." form into a ubiquitous-style FOR-every-step requirement with a precise citation format spec (`<path>::<Class>.<method>` or `<path>::<function>`).

One new criterion was added (the align_pose split), within the 3-new-criteria budget. No implementation details, colors, schemas, error strings, HTTP codes, or new technologies were introduced; existing project references (ModelWrapper, DecoderSplattingCUDA, Foundation_Model, config names, file paths) are preserved.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 13 to make it more precise and testable

**Prompt:**

### Requirement 13: Glossary and Abbreviations Document

**User Story:** As a reader encountering unfamiliar terms, I want a glossary of C3G-specific and adopted terminology, so that I can resolve terms without searching the internet.

#### Acceptance Criteria

1. THE Documentation_Set SHALL contain a file named `docs/11-glossary.md`.
2. THE `docs/11-glossary.md` file SHALL define each of the following terms: Gaussian Splatting, Spherical Harmonics (SH degree), Novel View Synthesis (NVS), Feed-forward reconstruction, Pose-free reconstruction, Feature distillation, Vision Foundation Model (VFM), VGGT, DUSt3R, CroCo, NoPoSplat, pixelSplat, MVSplat, DPT head, Gaussian adapter, View sampler, Context view, Target view, Epipolar line, PSNR, SSIM, LPIPS, mIoU.
3. THE `docs/11-glossary.md` file SHALL include at least one External_Reference for every term it defines, pointing to the paper, repository, or standards document that introduces or canonically defines the term.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 13 (Glossary and Abbreviations Document) to eliminate vagueness and close gaps:

Gaps addressed:
- P1 vague qualifier: "define each of the following terms" — criterion 2 now requires a ≥1-sentence definition stating what the term denotes and its role in the C3G context, giving testers an objective pass/fail rule.
- P1 derivable gap: the requirement title mentions "Abbreviations" but no criterion enforced abbreviation expansion. Added a new WHERE criterion (criterion 3) requiring expanded forms for parenthesised abbreviations (NVS, VFM, PSNR, SSIM, LPIPS, mIoU).
- P0 test blocker: criterion 3 demanded an External_Reference "for every term", but several listed terms are C3G-specific (Context view, Target view, Gaussian adapter, View sampler) and have no external canonical source. Amended criterion 4 to allow a link to the internal source file or Public_Symbol as a fallback, keeping the requirement testable and consistent with the rest of the spec (External_Reference defined in the Glossary).

Budget: 1 new criterion added (well within the 3-max), 2 amendments to existing criteria (criteria 2 and 3), criterion 1 preserved. No implementation details introduced; technology references already in the spec (C3G, VGGT, DUSt3R, CroCo, NoPoSplat, pixelSplat, MVSplat, DPT, etc.) are preserved.

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 14 to make it more precise and testable

**Prompt:**

### Requirement 14: Documentation Fidelity to Code

**User Story:** As a reader who trusts the documentation to describe current behavior, I want every code-level claim to match the actual source, so that I do not chase outdated instructions.

#### Acceptance Criteria

1. THE Documentation_Set SHALL reference only Public_Symbols that exist in the C3G_Repository at the time the documentation is written.
2. THE Documentation_Set SHALL reference only configuration keys that are defined in a file under `config/` or a dataclass under `src/config.py`, `src/model/model_wrapper.py`, or `src/dataset/`.
3. WHEN a Run_Recipe is included in the Documentation_Set, THE Run_Recipe SHALL use the exact command string pattern shown in `README.md` or a command that invokes `python -m src.main` with Hydra overrides that match existing keys.
4. IF a document makes a numerical claim (for example the 2K Gaussian count, the 224×224 input shape, the `NUM_SEMANTIC_CHANNELS` values), THEN THE document SHALL cite the source file and symbol from which the number is taken.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 14 (Documentation Fidelity to Code). Key gaps addressed:

1. Criterion 1: made "reference" testable by requiring exact case/spelling matching the defining `.py` file, and anchored "at the time the documentation is written" to the repo's main-branch HEAD at production time.
2. Criterion 2: defined "configuration keys" as Hydra dotted paths or dataclass field names, broadened the allowed dataclass scope to `src/model/` (since `ModelCfg`, `EncoderCfg`, etc. live there, not only in `model_wrapper.py`), and required exact-name matching.
3. Criterion 3: replaced vague "exact command string pattern" with "character for character inside its fenced code block" and tightened "match existing keys" to "resolves to a key defined under `config/` or in a dataclass parsed by `src/config.py`".
4. Added new criterion (was buried in the old #3): placeholders for checkpoint/dataset/weight paths and env vars must be explicitly marked and explained adjacent to the recipe (ensures Run_Recipes remain literal-but-reproducible).
5. Criterion 5 (was #4): specified citation format `<repo-relative file path>::<symbol>` and required the cited symbol to actually hold the claimed value, making the fidelity check executable.
6. Added new criterion: any quoted code/config/constant snippet must be character-for-character accurate and accompanied by its source path. Covers the fidelity risk for non-Public_Symbol content (YAML keys, constant literals, etc.).

Budget respected: 4 original criteria amended, 2 new criteria added (within the 3-new-criteria cap).

### Sub-Agent: Requirement Detailer

**Explanation:** Refining Requirement 15 to make it more precise and testable

**Prompt:**

### Requirement 15: Documentation Location and File Layout

**User Story:** As a reader discovering the documentation for the first time, I want all documentation files in a predictable location with predictable names, so that I can navigate them with file search alone.

#### Acceptance Criteria

1. THE Documentation_Set SHALL reside under the directory `docs/` at the root of the C3G_Repository.
2. WHERE the Documentation_Set resides under `docs/` at the root of the C3G_Repository, THE Documentation_Set SHALL use only Markdown files with the `.md` extension.
3. WHERE the Documentation_Set resides under `docs/` at the root of the C3G_Repository, THE top-level files of the Documentation_Set SHALL use the numbered prefix pattern `NN-slug.md` where `NN` is a two-digit number matching the reading order (for example `01-overview.md`, `02-getting-started.md`).
4. THE Module_Documents SHALL reside under `docs/modules/` as defined in Requirement 5.
5. THE Documentation_Set SHALL NOT modify any file outside the `docs/` directory.

Spec folder: /home/arunabh/C3G/.kiro/specs/repo-documentation/

**Response:**

Refined Requirement 15 to address 6 gaps: fixed WHERE→Ubiquitous pattern misuse on criteria 2 and 3; defined "top-level files" as direct children of docs/; specified slug character set (lowercase ASCII letters, digits, hyphens); added explicit exception for docs/README.md (Documentation_Index); added NN uniqueness/start/contiguity constraints as a new criterion; and fixed the subject of the non-modification rule to target the documentation change-set rather than the passive artifact. All changes are amendments plus one new criterion for NN sequencing, staying within budget.

All refinements complete. Now I'll update requirements.md with the detailed versions from each requirement-detailer.

Requirements are ready. Here's a summary of what the docs/ tree will cover:

**Orientation & onboarding**
- Overview of C3G as a feed-forward 2K-Gaussian framework, top-level directory tour, and pinned Python 3.11 / PyTorch 2.5.1 / CUDA 12.4 versions
- Documentation index at `docs/README.md` that always mirrors the reading order
- Getting-started page with verbatim install commands, pretrained-weight download steps, dataset-preparation pointers (pixelSplat, MVSplat, LSM, Feature 3DGS), and copy-pasteable Run_Recipes for every training and evaluation flow (including the per-model `NUM_SEMANTIC_CHANNELS` values)

**Understanding the system**
- Architecture doc with a Mermaid/ASCII diagram tracing `DataModule → Dataset* → ModelWrapper → Encoder{VGGT,NoPoSplat} → UnifiedGaussianAdapter → DecoderSplattingCUDA → Loss*`, typed with `BatchedExample` and `Gaussians`
- Configuration doc mapping every YAML under `config/` to its dataclass in `src/config.py`, including tables for the `cfg.train.reproj_model` → Foundation_Model selection
- Datasets & view-samplers doc covering RE10k, ScanNet, ScanNetPose, Replica, the four samplers, all `assets/evaluation_index_*.json`, and every shim
- Models doc documenting the encoder/decoder registries, the `Gaussians` dataclass with jaxtyping shapes, DPT heads, `gmae.py`'s `Transformer`/`InstillTransformer`, and the full `load_foundation_model.py` dispatch table

**Running & extending**
- Training doc covering `src/main.py`'s Hydra entry, `ModelWrapper.training_step`, every loss file (`LossMse`, `LossLpips`, `loss_point`, `loss_ss`, `loss_ssim`), and the three checkpoint-format branches
- Evaluation doc covering every `TestCfg` field, `compute_psnr`/`ssim`/`lpips`, the `PoseEvaluator` + `eval_pose.py` entry point, `EvaluationIndexGenerator`, ScanNet mIoU/accuracy flow, and the output directory layout
- Reading guide with 3–6 personas, interleaved external references (C3G paper, VGGT, NoPoSplat, DUSt3R, CroCo, 3DGS, pixelSplat, MVSplat, LSeg, DINOv2/3, CLIP), and a module-to-reference table
- Walkthroughs doc with three end-to-end call-trace walkthroughs (gaussian-head training, RE10K test step with `align_pose` branch, LSeg feature-head training)
- Glossary covering every C3G-specific and adopted term

**Quality gates**
- One Module_Document under `docs/modules/` for every Python package in `src/` (including `croco/`, `vggt/`, `clip/`, `distiller/`, etc.)
- Strict fidelity rules: every symbol, config key, command, and number must cite its source file
- Predictable file layout: `docs/NN-slug.md` for top-level docs, `docs/modules/*.md` for packages, no files touched outside `docs/`

You can now:
[Generate Tech Design](kiro-spec://create?featureName=repo-documentation&documentType=design)
[Generate Task List](kiro-spec://create?featureName=repo-documentation&documentType=tasks)
[Analyze Requirements](kiro-spec://spec?featureName=repo-documentation&action=analyze)

---

**User:**

Create the design for repo-documentation

---

**Assistant:**

On it.

---

**User:**



---

**Assistant:**

### Sub-Agent: Feature Requirements First Workflow

**Explanation:** Delegating design-document generation for the repo-documentation feature to the requirements-first workflow subagent, since the config already specifies workflowType=requirements-first and requirements.md is complete.

**Prompt:**

Create the design document for the "repo-documentation" feature spec.

Spec type: feature
Workflow type: requirements-first
Feature name: repo-documentation
Feature directory: /home/arunabh/C3G/.kiro/specs/repo-documentation/

The requirements document is already complete at /home/arunabh/C3G/.kiro/specs/repo-documentation/requirements.md and defines 15 requirements covering:
- R1 Repository Overview Document (docs/01-overview.md)
- R2 Documentation Index (docs/README.md)
- R3 Getting Started and Run Recipes (docs/02-getting-started.md)
- R4 Architecture Overview (docs/03-architecture.md)
- R5 Source Tree Module Documents (docs/modules/*.md, one per top-level src package)
- R6 Configuration System Document (docs/04-configuration.md)
- R7 Datasets and View Sampling Document (docs/05-datasets.md)
- R8 Models and Foundation Models Document (docs/06-models.md)
- R9 Training, Losses, and Optimization Document (docs/07-training.md)
- R10 Evaluation and Metrics Document (docs/08-evaluation.md)
- R11 Reading Guide with External References (docs/09-reading-guide.md)
- R12 Code Flow Walkthroughs (docs/10-walkthroughs.md)
- R13 Glossary and Abbreviations (docs/11-glossary.md)
- R14 Documentation Fidelity to Code (cross-cutting correctness properties)
- R15 Documentation Location and File Layout (docs/ root, NN-slug.md pattern)

Key context about the project (C3G):
- CVPR 2026 paper "Learning Compact 3D Representations with 2K Gaussians" (arXiv 2512.04021)
- Feed-forward framework predicting ~2K 3D Gaussians from unposed images
- Built on PyTorch Lightning + Hydra + CUDA splatting
- Backbones: CroCo, VGGT; Decoder: splatting_cuda; Foundation models: LSeg, DINOv2/3, VGGT, CLIP, MaskCLIP
- Source tree rooted at /home/arunabh/C3G
- Key directories: src/ (main.py, model/, dataset/, loss/, evaluation/, geometry/, misc/, visualization/, scripts/), config/ (Hydra), assets/ (eval indices), pretrained_weights/ (external)

This feature produces ONLY documentation Markdown files under docs/; it does NOT modify any runtime code. The design should therefore focus on:
- The information-extraction pipeline that reads the source tree and generates each document
- The document templates/structures that satisfy each requirement's acceptance criteria
- The fidelity-checking approach that satisfies R14 (claims match code)
- Cross-document consistency (index, reading guide, and module docs agreeing)
- How the NN-slug.md ordering (R15) and Module_Document naming rule (R5 Criterion 6) are enforced
- Mermaid/ASCII diagram approach for R4 Criterion 4
- Correctness properties suitable for property-based testing of the generation/validation logic

Please produce design.md at /home/arunabh/C3G/.kiro/specs/repo-documentation/design.md following the requirements-first design template, including: Overview, Architecture, Components and Interfaces, Data Models, Error Handling, Testing Strategy (with Property-Based Testing correctness properties), and any relevant sequence/flow diagrams.

**Response:**

Created /home/arunabh/C3G/.kiro/specs/repo-documentation/design.md for the requirements-first workflow's design phase.

The document contains all seven required sections: Overview, Architecture, Components and Interfaces, Data Models, Correctness Properties, Error Handling, and Testing Strategy. Two Mermaid diagrams illustrate the pipeline flow and the authoring sequence for a single document.

Key design decisions:
- A three-stage pipeline (SourceScanner → DocumentPlanner → Document Writers) feeds a FidelityValidator that enforces R14 by re-checking every CitationRef (SymbolCitation, LiteralCitation, QuoteCitation, YamlKeyCitation) against the RepoInventory.
- A pure function `derive_module_doc_path` mechanises R5.6; `nn_prefixes_are_contiguous` and `parse_nn_prefix` mechanise R15.3 and R15.4.
- Index/Reading-Guide consistency (R2.3) is guaranteed by both writers deriving their ordering from the same DocumentPlan list.
- An ExternalReferenceRegistry centralises every URL so R1.3, R8.8, and R11.4 reduce to tag-presence checks.

PBT prework was run via the prework tool. After property reflection, the design includes 20 consolidated correctness properties covering: NN-slug layout and contiguity, docs-path structural invariant, module-doc filename derivation, index-documents closure, index↔reading-guide ordering, symbol and quote fidelity, config-key fidelity, citation coverage, Run_Recipe validity, placeholder adjacency, version reconciliation, YAML set identity, package file/symbol coverage, upstream-reference presence, persona interleaving, dataset-adapter coverage, architecture dataclass/diagram coverage, and glossary abbreviation/hyperlink rules.

Testing strategy specifies Hypothesis + pytest with ≥200 iterations per property test, plus two integration tests that run against the actual C3G repo.

The design for the **repo-documentation** spec is ready. It lays out a pipeline for producing the full `docs/` tree: a Source Scanner builds a `RepoInventory` from `src/`, `config/`, `assets/`, `README.md`, and `requirements.txt`; a Document Planner assigns the fixed `NN-slug.md` ordering; per-document Writers render the Markdown; and a Fidelity Validator re-checks every citation against the source so documentation can't silently drift. The design captures 20 correctness properties (layout, fidelity, coverage, interleaving, glossary rules) and a Hypothesis-based testing strategy with unit, property, and integration tiers.

You can now:
[Generate Task List](kiro-spec://create?featureName=repo-documentation&documentType=tasks)

---

**User:**



---

**Assistant:**

Both prerequisites (requirements.md and design.md) exist. Delegating to the tasks phase.

### Sub-Agent: Feature Requirements First Workflow

**Explanation:** Delegating to the feature-requirements-first-workflow subagent with the tasks preset, since requirements.md and design.md already exist for the repo-documentation spec.

**Prompt:**

Create the tasks.md file for the "repo-documentation" feature spec. The spec is located at .kiro/specs/repo-documentation/. Both requirements.md and design.md already exist - please read them to understand the context, then generate a comprehensive tasks.md implementation plan that follows the design and satisfies the requirements.

Spec type: feature
Workflow: requirements-first
Feature name: repo-documentation

**Response:**

Created `.kiro/specs/repo-documentation/tasks.md` — an implementation plan for the C3G `repo-documentation` feature derived from the existing requirements.md (R1–R15) and design.md (20 correctness properties, `tools/docgen/` architecture).

Key decisions:
- **Language**: Python 3.11 (explicitly declared in the design's Testing Strategy — no userInput prompt needed since the design is not pseudocode).
- **Structure**: 20 top-level tasks, max two levels of hierarchy, 4 checkpoint tasks at incremental boundaries.
- **Build order**: data models → pure helpers → scanner → planner → writers (content writers before the `IndexWriter` and `ReadingGuideWriter` that depend on them) → filesystem writer → fidelity validator → CLI → actual `docs/` generation.
- **Property-based tests**: included as optional (`*`) sub-tasks placed next to the implementation they validate. Each of the 20 design properties maps to exactly one PBT sub-task, annotated with property number and the requirement clauses it validates (Properties 1–20 → Requirements from R1.5–R15.6).
- **R15.6 compliance**: all implementation tasks (1–18) stay inside `tools/docgen/`; only task 19 writes to `docs/`, so the change-set boundary is preserved until the generator is ready.
- **Coding-only scope**: no deployment, user testing, or performance-gathering tasks — only code, config, and automated tests.

The `.config.kiro` file was already present (created earlier in the workflow). The tasks.md is ready for the user to start executing via "Start task" in the UI.

The implementation plan for the repo documentation spec is ready. It's organized around building a `tools/docgen/` generator package incrementally: data models, pure helpers, the source scanner, the document planner, the writers in dependency order, the filesystem writer, the fidelity validator, the CLI entry points, and finally generating the real documentation tree.

The plan includes checkpoints at incremental boundaries so you can validate progress before moving to the next layer, and every correctness property from the design has an optional property-based test task next to the implementation it validates.

You can start executing by clicking "Start task" next to any task in `tasks.md`, or ask me to run all tasks.

---

**User:**

I don't care about building a documentation feature, I jsut want you to document the code.

---

**Assistant:**

Now let me read through the key source files to write proper documentation directly.

Now I have enough context to write comprehensive documentation. Let me create the docs.

### Sub-Agent: General Task Execution

**Explanation:** Delegating the creation of the full docs/ directory with comprehensive markdown documentation for the C3G repository. This is a large writing task that needs to read many source files and produce multiple documentation files.

**Prompt:**

Create comprehensive markdown documentation for the C3G repository under a `docs/` directory. The repo is at `/home/arunabh/C3G`. This is a CVPR 2026 paper codebase for "C3G: Learning Compact 3D Representations with 2K Gaussians" — a feed-forward framework that predicts ~2K 3D Gaussians from unposed images for novel view synthesis and 3D scene understanding.

Create the following files. Write them directly — no tooling, no generators, just the actual documentation markdown files.

## Key facts about the codebase (use these when writing):

**What it does**: Feed-forward model that takes unposed images → predicts ~2048 3D Gaussians → renders novel views via differentiable splatting. Also supports feature distillation from VFMs (LSeg, DINOv2, DINOv3, VGGT) for 3D scene understanding.

**Stack**: Python 3.11, PyTorch 2.5.1, CUDA 12.4, PyTorch Lightning, Hydra configs, gsplat 1.5.3

**Entry point**: `src/main.py` — Hydra-decorated `train()` function that builds Trainer, DataModule, ModelWrapper and dispatches to `trainer.fit` or `trainer.test`

**Encoders** (registered in `src/model/encoder/__init__.py`):
- `"noposplat"` → `EncoderNoPoSplat` (2-view, CroCo backbone)
- `"noposplat_multi"` → `EncoderNoPoSplatMulti` (multi-view)
- `"vggt"` → `EncoderVGGT` (VGGT backbone)

**Decoder**: `"splatting_cuda"` → `DecoderSplattingCUDA` (differentiable Gaussian rasterization)

**Losses** (registered in `src/loss/__init__.py`): `LossMse`, `LossLpips`. Additional: `loss_point.py`, `loss_ss.py`, `loss_ssim.py`

**Foundation models** (`src/model/load_foundation_model.py`):
- `vggt` / `vggt_tracking` → VGGT-1B, feature_dim=2048
- `dinov2_B` → dinov2_vitb14_reg, feature_dim=768
- `dinov2_L` → dinov2_vitl14_reg, feature_dim=1024
- `dinov3_L` → facebook/dinov3-vitl16-pretrain-lvd1689m, feature_dim=1024
- `dinov3_H` → facebook/dinov3-vith16plus-pretrain-lvd1689m, feature_dim=1280
- `dinov3_7B` → facebook/dinov3-vit7b16-pretrain-lvd1689m, feature_dim=4096
- `lseg` → local pretrained_weights/demo_e200.ckpt, feature_dim=512
- `maskclip` → mhamilton723/FeatUp maskclip, feature_dim=512

**Data types**:
- `BatchedExample` (TypedDict): `context` and `target` (each `BatchedViews` with extrinsics, intrinsics, image, near, far, index, overlap), plus `scene`
- `Gaussians` (dataclass): `means`, `covariances`, `harmonics`, `opacities`, `feature`

**Config structure** (`config/main.yaml`):
- defaults: model/encoder: noposplat, model/decoder: splatting_cuda, loss: [mse]
- Top-level keys: wandb, mode, data_loader, optimizer, checkpointing, train, test, seed, trainer
- Training configs under `config/training/`: gaussian_head, gaussian_head_multiview, feature_head_lseg, feature_head_dinov2_B, feature_head_dinov2_L, feature_head_dinov3_L, feature_head_vggt, feature_head_lseg_multiview
- Evaluation configs under `config/evaluation/`: re10k, re10k_multiview, scannet

**Datasets**: RE10K (training + eval), ScanNet (scene understanding eval), Replica (scene understanding eval)

**Key architectural insight**: The `training_step` runs encoder → decoder → losses. When `train.feature_rendering_loss > 0`, it also runs `forward_foundation_model` to get VFM features and computes a cosine-similarity feature loss between rendered Gaussian features and VFM features.

**Checkpoint loading** in `src/main.py` handles 3 formats:
1. `'model'` key → filter with `checkpoint_filter_fn`
2. `'state_dict'` key → strip `encoder.` prefix
3. Raw dict with `aggregator`/`point_head` keys → remap to backbone/dpt_head

**Test step**: When `test.align_pose=True`, runs iterative pose optimization (`test_step_align`) with rotation/translation deltas. Computes PSNR/SSIM/LPIPS. For scene understanding, uses `torchmetrics.JaccardIndex` and `Accuracy` with `len(labels)+1` classes.

## Files to create:

### 1. `docs/README.md` — Documentation Index
A table of contents linking to all other docs.

### 2. `docs/01-overview.md` — Repository Overview
What C3G is, the paper, key contributions, directory structure, tech stack, links to paper/project page/VGGT/NoPoSplat.

### 3. `docs/02-getting-started.md` — Getting Started
Installation (verbatim from README), pretrained weight downloads (VGGT model.pt, LSeg demo_e200.ckpt), HuggingFace checkpoints list, dataset preparation (RE10K via pixelSplat/MVSplat, ScanNet via LSM, Replica via Feature 3DGS), all training/evaluation commands from README, NUM_SEMANTIC_CHANNELS note.

### 4. `docs/03-architecture.md` — Architecture Overview
End-to-end data flow with a Mermaid diagram: DataModule → Dataset → ModelWrapper → Encoder → Gaussians → Decoder → rendered output → Loss. Explain BatchedExample and Gaussians types. Explain the feature distillation path. Explain encoder selection.

### 5. `docs/04-configuration.md` — Configuration System
Hydra setup, main.yaml structure, how training/evaluation configs compose (show defaults lists), the config-to-dataclass mapping table, the reproj_model → foundation model table.

### 6. `docs/05-datasets.md` — Datasets and View Sampling
Each dataset adapter, view samplers (all, arbitrary, bounded, evaluation), evaluation index JSONs, shims.

### 7. `docs/06-models.md` — Models and Foundation Models
Encoder variants, decoder, Gaussians dataclass, GaussianAdapter, DPT heads, Transformer/InstillTransformer, load_foundation_model dispatch, external references.

### 8. `docs/07-training.md` — Training and Losses
Entry point, training_step flow, loss classes, feature_rendering_loss, optimizer/trainer config, checkpoint loading.

### 9. `docs/08-evaluation.md` — Evaluation and Metrics
TestCfg fields, PSNR/SSIM/LPIPS metrics, PoseEvaluator, EvaluationIndexGenerator, ScanNet mIoU/Accuracy flow, output directory layout.

### 10. `docs/09-reading-guide.md` — Reading Guide
Three personas (Run the model, Understand the architecture, Extend a module) with ordered file lists interleaved with paper references. A module-to-reference table.

### 11. `docs/10-walkthroughs.md` — Code Flow Walkthroughs
Step-by-step traces of: (1) gaussian_head training step, (2) re10k test step, (3) feature_head_lseg training step. Each step cites file::function.

### 12. `docs/11-glossary.md` — Glossary
Define: Gaussian Splatting, SH, NVS, Feed-forward reconstruction, Pose-free reconstruction, Feature distillation, VFM, VGGT, DUSt3R, CroCo, NoPoSplat, pixelSplat, MVSplat, DPT head, Gaussian adapter, View sampler, Context view, Target view, Epipolar line, PSNR, SSIM, LPIPS, mIoU.

## Important instructions:
- Read source files as needed to get accurate information
- Include external reference links (arXiv papers, GitHub repos) throughout
- Use Mermaid diagrams where helpful
- Keep prose concise but thorough
- Cite source files when making claims about code behavior
- Make the docs useful for someone who has never seen this codebase before
- Write ALL files — don't skip any

**Response:**

Created all 12 documentation files under `docs/`:

1. `docs/README.md` — Documentation index with table of contents and quick links
2. `docs/01-overview.md` — Repository overview, contributions, tech stack, directory structure, related projects
3. `docs/02-getting-started.md` — Installation, weight downloads, dataset prep, all training/evaluation commands
4. `docs/03-architecture.md` — End-to-end data flow with Mermaid diagrams, core types (BatchedExample, Gaussians, DecoderOutput), encoder selection, feature distillation path
5. `docs/04-configuration.md` — Hydra setup, main.yaml structure, training/evaluation config composition, config-to-dataclass mapping, reproj_model → VFM table
6. `docs/05-datasets.md` — All dataset adapters (RE10K, ScanNet, Replica), view samplers (bounded, evaluation, all, arbitrary), evaluation index JSONs, data shims
7. `docs/06-models.md` — Encoder variants (VGGT, NoPoSplat), decoder, Gaussians dataclass, GaussianAdapter, DPT heads, Transformer/InstillTransformer, load_foundation_model dispatch
8. `docs/07-training.md` — Entry point, training_step flow with Mermaid diagram, loss classes, feature rendering loss, optimizer config, checkpoint loading (3 formats)
9. `docs/08-evaluation.md` — TestCfg fields, PSNR/SSIM/LPIPS metrics, pose alignment flow, PoseEvaluator, EvaluationIndexGenerator, ScanNet mIoU/Accuracy, output directory layout
10. `docs/09-reading-guide.md` — Three personas (Run, Understand, Extend) with ordered file lists and paper references, module-to-reference table
11. `docs/10-walkthroughs.md` — Step-by-step traces of gaussian_head training, re10k test, and feature_head_lseg training with file::function citations
12. `docs/11-glossary.md` — 25+ terms defined with external paper references

All documentation is based on actual source code reading and includes accurate file paths, function names, config keys, and external links.

Now let me read more source files to get accurate details for the documentation.

Now I have enough information to write all the documentation files. Let me start creating them.

Now let me verify all files were created:

All 12 files are created. Let me do a quick sanity check on a couple of them to make sure they rendered properly:

Done. I've created 12 documentation files under `docs/`:

| File | Content |
|------|---------|
| `docs/README.md` | Table of contents with links to all docs + quick links to paper/HuggingFace/VGGT |
| `docs/01-overview.md` | What C3G is, key contributions, tech stack, directory structure, related projects |
| `docs/02-getting-started.md` | Installation, weight downloads, dataset prep, all training/eval commands |
| `docs/03-architecture.md` | End-to-end data flow with Mermaid diagrams, core types, encoder selection |
| `docs/04-configuration.md` | Hydra setup, config composition, dataclass mapping, VFM selection table |
| `docs/05-datasets.md` | Dataset adapters, view samplers, eval indices, shims |
| `docs/06-models.md` | Encoders, decoder, Gaussians, GaussianAdapter, DPT heads, VFM loading |
| `docs/07-training.md` | Training step flow, losses, feature distillation loss, checkpoint formats |
| `docs/08-evaluation.md` | Test step, pose alignment, metrics, ScanNet mIoU, output layout |
| `docs/09-reading-guide.md` | Three personas (Run / Understand / Extend) with ordered reading lists |
| `docs/10-walkthroughs.md` | Step-by-step code traces for 3 scenarios with `file::function` citations |
| `docs/11-glossary.md` | 25+ terms defined with links to papers |

Start with `docs/README.md` for the table of contents, or jump straight to `docs/09-reading-guide.md` for a persona-based reading order tailored to your goal.

---
