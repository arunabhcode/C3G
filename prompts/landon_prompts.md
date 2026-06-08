# Landon's Prompts (C3G Project)

Prompts for the project.

## Project Foundation & SAM Integration

### 1.

look at this proposal and see what the gol of this project is: @/Users/landonchoy/Desktop/CS231N_Project_Proposal.pdf 

we want to use C3G + SAM. can you see where and what files int he repo this has begun to be integrated in

### 2. can you build the vanilla SAM forward pass from what we have here in the repo. Keep everything modular and have minimal code edits and new files. keep everything readable and contains file annotations

### 3.

build an eval pipeline for lerf-mask that gets only the images and masks and their corresponding camera poses. Get this from: [https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask) .

It should work with vanilla SAM and C3G+SAM.

don't tryt o verify code with python. the env isn't set up. I just wan the code set up

### 4.

can you implement boundarymIOU and warpmIOU metrics in the generic model wrapper and vanilla SAM as well.

boundaryIOU gets the mIOU of the boundary pixels of the segmented objects. warpmIOU gets the mIOU of a predicted mask projected into the camera pose of another predictd mask. Note that boundary mIOU is between the predicted mask and label mask and warpmIOU is between a predicted mask and another specifed predcited mask. leave that part todo implementatio for the warpmiou.

---

## Dataset Infrastructure (ScanNet, Replica, Modal Volumes)

---

## Dataset Infrastructure

### 5.

edit @src/dataset/download_scannet.py to be able to download 10 scenes and teh corresponding images and segmentation masks, camera poses that is necessary for @src/dataset/dataset_scannet.py and @src/dataset/dataset_scannet_pose.py.

and have this data get loaded into a modal volume called scannet

### 6.

can you edit @src/dataset/dataset_scannet.py and @src/dataset/dataset_scannet_pose.py to work for just 2D segmentation and not need depth or anything extraneous. Though keep the camera poses data.

do this if it doesn't already do this. 

do the same for @src/dataset/dataset_replica.py and @src/dataset/dataset_lerf_mask.py if they ren't alrady like that

### 7.

can you upload the @replica_semseg to modal and call in "replica" and add a dataset_replica_pose.py file that is similar to dataset_scannet_pose.py by being for 2D segmentation + pose data.

the things uploaded to modal volume should just be the pose information and the info neccessary for 2D segmentation training and inference. use @download_replica.py to do so

### 8. in @src/dataset/download_scannet.py , I only want the neccessary information for @src/dataset/dataset_scannet_2dseg.py . Nothing else.Take out the stuff required for @src/dataset/dataset_scannet.py and @src/dataset/dataset_scannet_pose.py. It should just be scenes not EVAL_SCENES. There should be 15 total scenes

### 9. can you change @src/dataset/replica_data/download_replica.py  , to have the same format for the replica volume in modal as @src/dataset/download_scannet.py . I want the _x, _y, and _cam. Though the cam might be a little different because it is a json. Maybe change it to be the same if possible and completely redo @src/dataset/dataset_replica_2dseg.py to accomodate this change and remember that download_replica is only for 2dseg and don't save anything else for other dataset files

### 10. I want @src/dataset/dataset_replica_2dseg.py to have the exact same logic as @src/dataset/dataset_replica_semseg.py , but be for the modal replica volume logic. can you do this?

### 11. do the same for @src/dataset/dataset_scannet_2dseg.py . it should have the exact same simplified logic but be for the scannet modal volume

### 12.

look at the modal volume scannet and separate the scenes into train, val, test sets. have 8 scenes be val and 24 scenes be test (they should be the last scenes in the modal volume in number).

the rest is train. have the wandb val metrics be done on the val set. the test set is something that will be run at a later time. I will have a script for that later.

### 13. can you chagne @src/dataset/dataset_scannet_2dseg.py and @src/dataset/dataset_replica_2dseg.py to use that view sampler like the other datasets? then have in the configs for the @src/inference/modal_train_c3g_sam.py to specified value set to 10 for now (for the gap)

### 14.

Modify DatasetScannet2dSeg to support fixed multi-target scene batching.

note that below M =2 because I am using the 2-view gaussain decoder

Goal:

- Each yielded sample should contain:
  - fixed N context views (already exists via view_sampler.num_context_views, typically 2)
  - fixed M target views (new config parameter)
- This enables batching multiple scenes together with consistent tensor shapes.

Required changes:

1. Add config field:
  num_target_views: int = 1

to Scannet2dSegCfg.

1. In **iter**:

- Remove the loop:
  for target_idx in target_indices:
- Instead, sample exactly num_target_views target indices from target_indices.
- Skip the scene if len(target_indices) < num_target_views.

Example:
    perm = torch.randperm(len(target_indices))[:self.cfg.num_target_views]
    sampled_target_indices = [target_indices[i] for i in perm]

1. Build idxs using:
  idxs = list(context_indices) + sampled_target_indices
2. Update target metadata handling:

- target_frame_id -> target_frame_ids
- near/far should use num_target_views instead of 1
- target["index"] should contain all sampled target frame ids

1. Ensure output tensor shapes become:
  context.image -> [N, 3, H, W]
    target.image  -> [M, 3, H, W]
    target.label  -> [M, H, W]
2. Keep batching compatible with standard PyTorch collation so batch_size > 1 works across scenes.

---

## Modal Training Pipeline

### 15. set up the modal inference interface for the vanilla SAM in the repo andthe C3G SAM pipeline. only those two things. nothing else

### 16. can you add a modal training script for the C3G-F feature decoder that goes into the SAM decoder for the C3G SAM with modal

### 17.

can you update @src/inference/modal_train_c3g_sam.py to work for both scannet and replica per @docs/prompted_training_modal.md. 

it should work wiht the 2dseg datasets ofor scannet and replica

### 18. update the rest of the files in @inference accordingly as well and add a test option that tests one image (smoke test)

### 19.

do the direct upsample for the scannet dataset in the c3g-sam pipeline with this dual-resolution. it should be relatively minimal as only the sam encoder has this func right? 

for the replica dataset, have a raiseImplementedError to say that the direct upsampling to 1024x1024 hasn't be implemented yet. those are the only two datasets fyi

### 20. can you freeze the vggt backbone and the Q projects and K projections in the Instill Trasnformre for the @src/inference/modal_train_c3g_sam.py

### 21.

Implement mixed precision training with bfloat16 for the C3G-SAM pipeline.

Requirements:

1. Wrap model forward passes in torch.autocast(device_type="cuda", dtype=torch.bfloat16).
2. Do NOT cast model weights manually with .half() or .bfloat16().
3. Keep optimizer steps, optimizer states, and gradient updates in normal fp32 behavior.
4. Compute segmentation loss in fp32 for stability by casting logits to float before the criterion:
  loss = criterion(logits.float(), labels)
5. If VGGT and SAM image encoder are frozen, run them under both torch.no_grad() and autocast:
  with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
       vggt_feats = vggt(images)
       sam_feats = sam_encoder(images)
   Then detach outputs before passing to trainable modules.
6. Run trainable modules, including Gaussian decoder/encoder, fusion transformer, V projections, FFNs, output projection, and segmentation head, under autocast.
7. Use bf16 only on CUDA. Add a config flag such as train.mixed_precision: "bf16" / "fp32".
8. Ensure validation/inference also uses autocast for forward passes, but compute metrics from fp32 logits or postprocessed outputs.
9. Do not use GradScaler for bf16. Only add GradScaler if fp16 support is explicitly enabled later.
10. Add a sanity check/log showing whether mixed precision is enabled and which dtype is being used.

Expected training structure example:

with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
    logits = model(batch)

loss = criterion(logits.float(), labels)
loss.backward()
optimizer.step()

### 22.

can you add a modal script that take in @scripts/precompute_sam_features.py in @src/modal_infra/ and runs it on a modal gpu and saves the precomputed features to precompute_sam_features/{dataset} in modal?

the volume should be called precompute_sam_features

and there shoul dbe a folder for replica and scannet for when they are precomputed

### 23. can you change @src/modal_infra/modal_train_c3g_sam.py to do the sam precomputed features instead and that whole pipeline? and then have those features come fromt precompute_sam_features modal volume. for scannet, it is in the precompute_sam_features/scannet. look at that modal volume. the actual training script should be similar to @src/main.py , but just run through modal

### 24.

@modal_train_c3g_sam.py @modal_sam_common.py 

can you simplify these two scripts significantly to just use @src/main.py with the modal volumes hooked up? and the image built with uv pip install? also take out all the hydra overrides and have the configs only rely on the yaml files.

also change @config/training/feature_head_sam_precomputed.yaml to to modal paths where necessary

### 25. allow for the cli for @src/modal_infra/modal_train_c3gsam.py to be able to choose between the distillation and non-distillation experiment. that should be the only hyrad override.

### 26. okay, can you include that logic using the precomputed sam features using everything that you mentioned for what needs to be wired up for the prompted training? do this in minimal files changed and code

### 27. can you have @src/modal_infra/modal_train_c3gsam.py prompted mode to get the val loss as well and only save the checkpoints based on that, not iou at all

### 28. through the @src/main.py pipeline, have the prompted sam segmentation train on an unfrozen architecture all around except for the sam encoder and decoder

### 29. okay, can you have the same prompted mode for vanilla SAM in the codebase?

---

## Modal Inference

### 30. i want @src/modal_infra/modal_vanilla_sam.py to run on the test split of scannet and all of replica. don't choose between the two datasets. let me know if you are confused on what the test set of scannet is

### 31. alright for @src/modal_infra/modal_vanilla_sam.py , generate the binary masks for all the classes with the pixel value as the pred number for that view. in the modal vanilla-sam-outputs modal volume, have the view be a folder in the scene with the binary masks in that folder

---

## Eval Pipeline

### 32. can you implement @src/modal_infra/modal_eval_c3gsam.py that looks at the c3g-weights volume and evals like @src/modal_infra/modal_eval_sam.py but using the distillation-base.ckpt from the c3g-weights modal volume. i also want it to produce the same visualizations as during training (create 5 total. randommly sampled with a seed set). the evals should be on the test set

### 33. yes, please refactor to do a batched sam decoder

### 34. can u just combine the 2? the eval for sam and c3g-sam should be the same for modal

### 35.

for the eval 

make the vanilla SAM and C3G-SAM mask export evaluation as controlled as possible. Fix only issues that are actually present.

Goal:

The two backends should differ only in feature source:

- Vanilla: SAM image encoder features from RGB
- C3G-SAM: rendered C3G features

Everything else should be shared or equivalent:

- same frames/classes
- same prompt source
- same prompt coordinate convention
- same multimask selection rule
- same upsampling/postprocessing rule
- same final saved mask resolution

Specific checks/fixes:

1. Prompt generation

Check whether vanilla and C3G prompts are generated in different coordinate frames.

Currently vanilla may use coord_frame="original" while C3G may use coord_frame="sam".

If present, refactor so prompts are first generated once in original label/image pixel coordinates for both backends.

Create a helper like:

```
class_prompts_from_label_original(...)
```

Then add a shared helper to transform original coordinates into the coordinate system expected by the mask decoder.

Do not let C3G manually scale prompts with a different rule than vanilla unless it is mathematically identical to SAM’s ResizeLongestSide transform.

1. Mask selection

Check whether either backend uses ground-truth IoU to choose among multimasks.

If present, remove it.

Both backends should choose multimask via the mask decoder/SAM predicted IoU head:

```
best_idx = iou_predictions.argmax(...)
```

If both already do this, leave it unchanged.

1. Upsampling/postprocessing

Check whether vanilla uses SAM postprocess_masks while C3G uses F.interpolate.

If present, make the postprocessing rule consistent.

Preferred simple controlled fix:

- Export logits from both pipelines at decoder resolution.
- Upsample logits to the label-map size using the same function:
    F.interpolate(logits, size=label_size, mode="bilinear", align_corners=False)
- Threshold both the same way:
    mask = logits_up > 0

Create a shared helper:

```
upsample_logits_to_label_mask(logits, label_size) -> np.ndarray
```

Use this helper for both vanilla and C3G.

1. Vanilla SAM export path

If the vanilla export currently calls full sam_forward and receives already postprocessed masks, modify it so it can return low-res logits and iou_predictions.

Then choose best multimask using iou_predictions and apply the shared upsampling helper.

Important:

Do not save SAM’s already-postprocessed mask if C3G is using manual bilinear upsampling.

1. C3G export path

Keep C3G using rendered features, but make sure:

- point prompts are transformed using the same helper/convention as vanilla
- selected multimask uses predicted iou_predictions
- logits are upsampled with the shared helper
- final PNG size matches the label map exactly

1. Output resolution

Before saving any mask, assert:

```
mask.shape == label_np.shape[:2]
```

or equivalent using the GT mask shape.

1. Manifest/debug metadata

Add fields to the manifest documenting the controlled choices:

```
"controlled_eval": true

"prompt_source": "label_centroid_original_pixels"

"multimask_selection": "predicted_iou_head"

"logit_upsampling": "bilinear_align_corners_false_to_label_size"

"threshold": "logits>0"
```

1. Avoid changing unrelated training/model code.

This should be an evaluation/export-only change.

1. Add comments near the controlled helpers explaining:

The purpose is to make vanilla SAM and C3G-SAM differ only by feature source, not by prompt generation, mask choice, or upsampling.

After changes, verify by running a small export with limit_frames=1 and checking:

- vanilla and C3G save the same class_id filenames for the same frame
- all saved PNGs have the exact same H×W as the label map
- neither backend uses GT IoU to select a mask
- both use the same helper for final logit upsampling/thresholding

### 36.

implement @src/modal_infra/modal_get_scores.py that gets the scores between the predictions from the vanilla-sam-outputs and the scannet and replica dataset val and test set labels.

then it should get the boundary miou which gets the pixels of the boundary of the segmented classes and gets iou over those

then it should get the warped iou. the warped iou should be the reporjecgtion from one frame to another in frame order. for ex. the first frame avaiable and second frame (and vice versa) and second and third (and vice versa) but not first and third

### 37. there should only be modal_get_scores. no vanilla_sam_scores. it should take in the name of the experiment whether its c3gsam or sam. in both cases, the iou score metrics gathered should be the same for both preds. the preds are already generated in vanilla-sam-outputs and c3g-sam-eval-outputs. you jsut have to get them and compare them to the labels in the test set in scannet and replica

### 38.

The current scorer does NOT compute standard semantic segmentation IoU.

Right now `_shared_class_iou_scores()` computes:

1. Binary IoU separately for each class in each frame.
2. Appends all class IoUs into a list.
3. Averages them at the end.

This is effectively a mean-per-class binary IoU and not the standard semantic segmentation IoU used in most segmentation papers.

Additionally, classes are currently filtered using:

```
shared_classes = gt_classes & _classes_in_dense_map(pred_dense)
```

which excludes GT classes that were completely missed by the prediction. Those missed classes should contribute IoU = 0, not be dropped from evaluation.

Please change the implementation to compute dataset-level global IoU.

Desired behavior:

For every GT class present in a frame:

```
pred_bin = pred_dense == class_id
gt_bin   = gt_dense == class_id
```

Accumulate:

```
total_intersection += (pred_bin & gt_bin).sum()
total_union += (pred_bin | gt_bin).sum()
```

across ALL frames and ALL classes.

Final IoU should be:

```
global_iou = total_intersection / total_union
```

where the division is performed only once after all frames/classes have been processed.

Important:

- Do NOT restrict scoring to classes present in both prediction and GT.
- Use GT-present classes as the scoring set.
- If a GT class is completely missing from prediction:
intersection = 0
union > 0
IoU contribution should therefore be 0.
- Missing classes should not be silently excluded.

For boundary IoU:

Apply the same principle.

Instead of averaging per-class boundary IoUs, accumulate:

```
total_boundary_intersection
total_boundary_union
```

across all GT-present classes and frames, then compute:

```
global_boundary_iou =
    total_boundary_intersection / total_boundary_union
```

at the end.

For warp IoU:

Leave the current implementation unchanged for now.

Implementation suggestion:

Create a helper that returns raw counts:

```
intersection
union
boundary_intersection
boundary_union
```

for a given class.

Then aggregate counts throughout scoring and compute the final ratios once at the dataset level.

The final reported metrics should represent:

```
IoU           = global pixel IoU
boundary_iou  = global boundary IoU
```

rather than mean-per-class IoU.

I still want per scene iou saved to the modal volume

### 39. cahnge @src/modal_infra/modal_eval_masks.py to do dense mask prediciton

### 40. change @src/modal_infra/modal_eval_masks.py and @src/modal_infra/modal_get_scores.py pipelines to do this logit overlap resolution by overlap instead. thi also means you have to save the logits per bianry mask created@/Users/landonchoy/Desktop/logit_overlap_resolution_instructions.pdf

### 41. edit @src/modal_infra/modal_eval_masks.py to run distillation-diff_learnable_tokens.ckpt, which is also a c3gsam, but call it c3gsam-dft in the cli for the name, but just load those weights instead. it should save to c3g-sam-dft-eval-outputs volume

### 42. to @src/modal_infra/modal_eval_masks.py , can you add the ema-nomag.ckpt, which is the exact same as the c3g model loading but is a different ckpt. also update @src/modal_infra/modal_get_scores.py with the new ckpt and modal volume accordingly

---

## Architecture Documentation

### 43. can you write @docs/arch-details.md  with all these detials about the architecture and the diagram as well

### 44. what is teh math of the loss in @src/main.py for the distillation and prompted c3g-sam? include that in a seciton in @docs/arch-details.md

### 45. write this to @docs/arch-details.md in a section. also include how often the validation was done and ckpting and how that worked. this is for the prompted and distillationt training. be concise but incldue all the necessary information

---

## Visualization & Analysis

### 46.

implement @src/modal_infra/visualizations.py to use @form1 to generate 4 figures. The first should be just the plot of the loss:total.csv over the steps

the secnd should be 3 plots. on the left, there should be rgb loss which is the sum of the rgb_lpips and rgb_mse values at each step, in the middle, it should be the feature loss, and on the right it should be the seg loss.

output the resulting 2 figures to the form1 folder

### 47.

can you implement @src/modal_infra/seg_viz.py. refer to @src/modal_infra/modal_get_scores.py to see what modal volumes contain the eval outputs and what they are called. then go into each one and take two consecutive frames from the same random scene in both scannet and replica. Then get the GT mask for those frames.

the color code the classes to be the same color for the same classes across all dense masks. and save a figure for each of the modal eval volumes and the GT that contains the two segmented masks that are colored. have black be the background

### 48.

change @src/modal_infra/seg_viz.py to write to c3gsam_results/seg_results. not seg_results.

then have a local entrypoint for the file (not through modal) takes the existing files (if the exist) and create a figure table with scannet pictures in the left and replica on the right

then have a third column on the far left that says the method.

### 49.

save a plot for only GT, SAM, C3G-SAM,

then save another plot for GT, C3G-SAM (renamed to EMA + mag head + zero ad), C3G-SAM (DFT) (renamed to EMA + mag head + up proj), C3G-SAM (no mag head) (renamed to no EMA + no mag head), C3G-SAM  (EMA no mag head) (renamed to EMA + no mag head

### 50. in @src/modal_infra/loss_plots.py , can you also plot @c3gsam_results/form2/ in a figure with total loss and then another figure with the feature loss on the left and the feature magnitude loss on the right

### 51.

implement @src/modal_infra/data_examples.py that generates examples of the data inlcuding input and output mask. Refer to @src/modal_infra/seg_viz.py for formatting and how to do it. top row should be actual image, bottom should be GT mask. no need to label the rows, only the dataset. have 2 examples per dataset

it should be a 2x4 plot

---

