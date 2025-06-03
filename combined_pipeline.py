#!/usr/bin/env python3
"""
Combined Pipeline: FLUX Image Generation + 4D-Humans SMPL Extraction
Integrated into infinigen-affordance repository

This script combines:
1. FLUX image generation (person on bed)
2. 4D-Humans pose and shape estimation

Usage:
python combined_pipeline.py --bed_image images/bed3.png --mask_image images/mask.jpg --prompt "your prompt"
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import json
from typing import Optional

# Add the local 4D-Humans directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '4D-Humans'))

# FLUX/Image Generation imports
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

# 4D-Humans imports
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from yacs.config import CfgNode
import pyrender
import trimesh

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

class CombinedPipeline:
    def __init__(self, 
                 checkpoint=DEFAULT_CHECKPOINT,
                 detector='vitdet',
                 device=None):
        """
        Initialize the combined pipeline
        
        Args:
            checkpoint: Path to 4D-Humans checkpoint
            detector: Detector type ('vitdet' or 'regnety')
            device: Torch device to use
        """
        print("Initializing Combined Pipeline...")
        
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize FLUX pipeline for image generation
        print("Loading FLUX image generation pipeline...")
        self.flux_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Initialize 4D-Humans pipeline
        print("Loading 4D-Humans model...")
        self.model, self.model_cfg = load_hmr2(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup detector
        print(f"Loading {detector} detector...")
        self._setup_detector(detector)
        
        # Setup renderer - optimized for 256x256 resolution
        self.renderer = Renderer(self.model_cfg, faces=self.model.smpl.faces)
        
        print("Pipeline initialization complete!")
    
    def _setup_detector(self, detector_type):
        """Setup the human detector"""
        if detector_type == 'vitdet':
            from detectron2.config import LazyConfig
            import hmr2
            cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif detector_type == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def generate_image(self, bed_image_path, mask_image_path, prompt, 
                      height=800, width=800, guidance_scale=30, 
                      num_inference_steps=50, seed=30):
        """
        Generate an image of a person on a bed using FLUX
        
        Args:
            bed_image_path: Path to the empty bed image
            mask_image_path: Path to the mask image
            prompt: Text prompt for generation
            height, width: Output image dimensions
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image of the generated result
        """
        print(f"Generating image with prompt: '{prompt}'")
        
        # Load input images
        bed_image = load_image(bed_image_path)
        mask_image = load_image(mask_image_path)
        
        # Generate image
        generator = torch.Generator("cpu").manual_seed(seed)
        generated_image = self.flux_pipe(
            prompt=prompt,
            image=bed_image,
            mask_image=mask_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        
        print("Image generation complete!")
        return generated_image
    
    def extract_smpl_from_image(self, image, batch_size=8):
        """
        Extract SMPL parameters from an image using 4D-Humans
        
        Args:
            image: PIL Image or cv2 image (BGR format)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing extracted SMPL parameters and metadata
        """
        print("Extracting SMPL parameters...")
        
        # Convert PIL to cv2 if needed
        if hasattr(image, 'convert'):  # PIL Image
            img_cv2 = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:  # Already cv2 format
            img_cv2 = image
        
        # Detect humans in image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        
        if len(boxes) == 0:
            print("No humans detected in the image!")
            return None
        
        print(f"Detected {len(boxes)} person(s) in the image")
        
        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_results = []
        
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
            
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            batch_size_actual = batch['img'].shape[0]
            for n in range(batch_size_actual):
                person_id = int(batch['personid'][n])
                
                # Extract SMPL parameters
                vertices = out['pred_vertices'][n].detach().cpu().numpy()
                camera_translation = pred_cam_t_full[n]
                smpl_params = {
                    'global_orient': out['pred_smpl_params']['global_orient'][n].detach().cpu().numpy(),
                    'body_pose': out['pred_smpl_params']['body_pose'][n].detach().cpu().numpy(),
                    'betas': out['pred_smpl_params']['betas'][n].detach().cpu().numpy(),
                } if 'pred_smpl_params' in out else None
                
                result = {
                    'person_id': person_id,
                    'vertices': vertices,
                    'camera_translation': camera_translation,
                    'smpl_params': smpl_params,
                    'bounding_box': boxes[person_id] if person_id < len(boxes) else None,
                    'focal_length': float(scaled_focal_length),
                }
                
                all_results.append(result)
        
        print(f"SMPL extraction complete! Processed {len(all_results)} person(s)")
        return all_results
    
    def save_results(self, generated_image, smpl_results, output_dir, base_filename):
        """
        Save all results to the output directory
        
        Args:
            generated_image: PIL Image of the generated result
            smpl_results: List of SMPL extraction results
            output_dir: Output directory path
            base_filename: Base filename for saving files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save generated image
        image_path = os.path.join(output_dir, f"{base_filename}_generated.png")
        generated_image.save(image_path)
        print(f"Saved generated image: {image_path}")
        
        if smpl_results is None or len(smpl_results) == 0:
            print("No SMPL results to save")
            return
        
        # Convert PIL to cv2 for rendering
        img_cv2 = cv2.cvtColor(np.array(generated_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Save individual person results
        for i, result in enumerate(smpl_results):
            person_id = result['person_id']
            vertices = result['vertices']
            cam_t = result['camera_translation']
            
            # Resize image to 256x256 for better mesh overlay alignment with training resolution
            original_height, original_width = img_cv2.shape[:2]
            img_cv2_resized = cv2.resize(img_cv2, (256, 256))
            
            # Get image dimensions
            img_height, img_width = img_cv2_resized.shape[:2]
            
            # Prepare image tensor with correct normalization
            img_tensor = torch.from_numpy(img_cv2_resized.transpose(2, 0, 1)).float() / 255.0
            
            # Render the person with optimal resolution
            regression_img = self.renderer(
                vertices,
                cam_t,
                img_tensor,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            
            # Save rendered result
            render_path = os.path.join(output_dir, f"{base_filename}_person_{person_id}_render.png")
            cv2.imwrite(render_path, 255 * regression_img[:, :, ::-1])
            print(f"Saved person {person_id} render: {render_path}")
            
            # Also save side view
            white_img = (torch.ones(3, img_height, img_width) * torch.tensor([1, 1, 1]).view(3, 1, 1)).float()
            side_view_img = self.renderer(
                vertices,
                cam_t,
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True
            )
            side_render_path = os.path.join(output_dir, f"{base_filename}_person_{person_id}_side_view.png")
            cv2.imwrite(side_render_path, 255 * side_view_img[:, :, ::-1])
            print(f"Saved person {person_id} side view: {side_render_path}")
            
            # Save top view
            top_view_img = self.renderer(
                vertices,
                cam_t,
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                top_view=True
            )
            top_render_path = os.path.join(output_dir, f"{base_filename}_person_{person_id}_top_view.png")
            cv2.imwrite(top_render_path, 255 * top_view_img[:, :, ::-1])
            print(f"Saved person {person_id} top view: {top_render_path}")
            
            # Save mesh
            mesh_path = os.path.join(output_dir, f"{base_filename}_person_{person_id}.obj")
            camera_translation = cam_t.copy()
            tmesh = self.renderer.vertices_to_trimesh(vertices, camera_translation, LIGHT_BLUE)
            tmesh.export(mesh_path)
            print(f"Saved person {person_id} mesh: {mesh_path}")
            
            # Save SMPL parameters
            if result['smpl_params'] is not None:
                params_path = os.path.join(output_dir, f"{base_filename}_person_{person_id}_smpl.json")
                # Convert numpy arrays to lists for JSON serialization
                smpl_params_json = {}
                for key, value in result['smpl_params'].items():
                    smpl_params_json[key] = value.tolist()
                
                save_data = {
                    'smpl_params': smpl_params_json,
                    'camera_translation': cam_t.tolist(),
                    'focal_length': result['focal_length'],
                    'bounding_box': result['bounding_box'].tolist() if result['bounding_box'] is not None else None
                }
                
                with open(params_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"Saved person {person_id} SMPL params: {params_path}")
    
    def run_full_pipeline(self, bed_image_path, mask_image_path, prompt,
                         output_dir="output", base_filename=None,
                         **generation_kwargs):
        """
        Run the complete pipeline: generation + SMPL extraction
        
        Args:
            bed_image_path: Path to empty bed image
            mask_image_path: Path to mask image
            prompt: Generation prompt
            output_dir: Output directory
            base_filename: Base filename for outputs
            **generation_kwargs: Additional arguments for image generation
            
        Returns:
            Tuple of (generated_image, smpl_results)
        """
        start_time = time.time()
        
        if base_filename is None:
            base_filename = f"result_{int(time.time())}"
        
        print("=" * 70)
        print("INFINIGEN-AFFORDANCE: Image Generation + 4D-Humans Pipeline")
        print("=" * 70)
        
        # Step 1: Generate image
        print("\nSTEP 1: Generating image...")
        generated_image = self.generate_image(
            bed_image_path, mask_image_path, prompt, **generation_kwargs
        )
        
        # Step 2: Extract SMPL parameters
        print("\nSTEP 2: Extracting SMPL parameters...")
        smpl_results = self.extract_smpl_from_image(generated_image)
        
        # Step 3: Save all results
        print("\nSTEP 3: Saving results...")
        self.save_results(generated_image, smpl_results, output_dir, base_filename)
        
        end_time = time.time()
        print(f"\nPipeline completed in {end_time - start_time:.2f} seconds")
        print("=" * 70)
        
        return generated_image, smpl_results


def main():
    parser = argparse.ArgumentParser(description='Infinigen-Affordance: Combined Image Generation + 4D-Humans Pipeline')
    
    # Input arguments
    parser.add_argument('--bed_image', type=str, default='images/bed3.png', help='Path to empty bed image (relative to infinigen-affordance)')
    parser.add_argument('--mask_image', type=str, default='images/mask.jpg', help='Path to mask image (relative to infinigen-affordance)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory (relative to infinigen-affordance)')
    parser.add_argument('--base_filename', type=str, default=None, help='Base filename for outputs')
    
    # Generation parameters
    parser.add_argument('--height', type=int, default=800, help='Generated image height')
    parser.add_argument('--width', type=int, default=800, help='Generated image width')
    parser.add_argument('--guidance_scale', type=float, default=30, help='Guidance scale for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=30, help='Random seed')
    
    # 4D-Humans parameters
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to 4D-Humans checkpoint')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Detector type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for SMPL extraction')
    
    args = parser.parse_args()
    
    # Change to infinigen-affordance directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize pipeline
    pipeline = CombinedPipeline(
        checkpoint=args.checkpoint,
        detector=args.detector
    )
    
    # Run pipeline
    generated_image, smpl_results = pipeline.run_full_pipeline(
        bed_image_path=args.bed_image,
        mask_image_path=args.mask_image,
        prompt=args.prompt,
        output_dir=args.output_dir,
        base_filename=args.base_filename,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed
    )
    
    print(f"\nResults saved to: {args.output_dir}")
    if smpl_results:
        print(f"Extracted SMPL parameters for {len(smpl_results)} person(s)")


if __name__ == '__main__':
    main()
