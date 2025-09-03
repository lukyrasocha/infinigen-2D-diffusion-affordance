#!/usr/bin/env python3
"""
Example usage of the Infinigen-Affordance Combined Pipeline

This demonstrates how to generate images of people on beds and extract SMPL parameters.
"""

from combined_pipeline import CombinedPipeline

def main():
    # Initialize the pipeline
    pipeline = CombinedPipeline(detector='vitdet')
    
    print("\n" + "="*50)
    print("EXAMPLE 1: Person lying comfortably on bed")
    print("="*50)
    
    generated_image, smpl_results = pipeline.run_full_pipeline(
        bed_image_path="images/bed3.png",
        mask_image_path="images/mask.jpg",
        prompt="a person with blue jeans and white t-shirt lying comfortably on a bed",
        output_dir="output",
        base_filename="example1_lying",
        height=800,
        width=800,
        guidance_scale=30,
        num_inference_steps=50,
        seed=42
    )
    
    print("\n" + "="*50)
    print("EXAMPLE 2: Person sitting on bed")
    print("="*50)
    
    generated_image2, smpl_results2 = pipeline.run_full_pipeline(
        bed_image_path="images/bed3.png",
        mask_image_path="images/mask.jpg",
        prompt="a woman in elegant dress sitting gracefully on a bed reading a book",
        output_dir="output",
        base_filename="example2_sitting",
        height=800,
        width=800,
        guidance_scale=30,
        num_inference_steps=50,
        seed=123
    )
    
    print("PIPELINE COMPLETE")
    print("="*50)
    
    total_people = 0
    if smpl_results:
        total_people += len(smpl_results)
        print(f"✅ Example 1: Generated image and extracted SMPL for {len(smpl_results)} person(s)")
    
    if smpl_results2:
        total_people += len(smpl_results2)
        print(f"✅ Example 2: Generated image and extracted SMPL for {len(smpl_results2)} person(s)")

if __name__ == '__main__':
    main()
