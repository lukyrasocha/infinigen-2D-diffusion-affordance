#!/usr/bin/env python3
"""
Batch processing script for Infinigen-Affordance Combined Pipeline

Process multiple prompts and generate comprehensive datasets for affordance research.
"""

import os
import json
from combined_pipeline import CombinedPipeline

def main():
    # Initialize pipeline once for efficiency
    print("🚀 Initializing Infinigen-Affordance Pipeline for Batch Processing...")
    pipeline = CombinedPipeline(detector='vitdet')
    
    # Define batch scenarios for affordance research
    scenarios = [
        {
            "name": "lying_casual",
            "prompt": "a person with blue jeans and white t-shirt lying comfortably on a bed",
            "seed": 42
        },
        {
            "name": "sitting_reading", 
            "prompt": "a woman in elegant dress sitting gracefully on a bed reading a book",
            "seed": 123
        },
        {
            "name": "lying_business",
            "prompt": "a man in business suit lying on a bed reading a book",
            "seed": 456
        },
        {
            "name": "sitting_casual",
            "prompt": "a person in casual clothes sitting cross-legged on a bed with a laptop",
            "seed": 789
        },
        {
            "name": "lying_sleep",
            "prompt": "a person in pajamas lying on a bed in a sleeping position",
            "seed": 321
        },
        {
            "name": "sitting_edge",
            "prompt": "a person sitting on the edge of a bed putting on shoes",
            "seed": 654
        }
    ]
    
    # Process each scenario
    batch_results = []
    total_people = 0
    
    print(f"\n📋 Processing {len(scenarios)} scenarios for affordance dataset...")
    print("=" * 70)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎬 Scenario {i}/{len(scenarios)}: {scenario['name']}")
        print(f"💭 Prompt: {scenario['prompt']}")
        
        try:
            generated_image, smpl_results = pipeline.run_full_pipeline(
                bed_image_path="images/bed3.png",
                mask_image_path="images/mask.jpg", 
                prompt=scenario['prompt'],
                output_dir=f"batch_output",
                base_filename=f"scenario_{i:02d}_{scenario['name']}",
                height=800,
                width=800,
                guidance_scale=30,
                num_inference_steps=50,
                seed=scenario['seed']
            )
            
            # Track results
            scenario_result = {
                "scenario_id": i,
                "name": scenario['name'],
                "prompt": scenario['prompt'],
                "seed": scenario['seed'],
                "success": smpl_results is not None,
                "num_people": len(smpl_results) if smpl_results else 0,
                "files_generated": []
            }
            
            if smpl_results:
                total_people += len(smpl_results)
                for result in smpl_results:
                    person_id = result['person_id']
                    base_name = f"scenario_{i:02d}_{scenario['name']}_person_{person_id}"
                    scenario_result["files_generated"].extend([
                        f"{base_name}_render.png",
                        f"{base_name}_side_view.png", 
                        f"{base_name}_top_view.png",
                        f"{base_name}.obj",
                        f"{base_name}_smpl.json"
                    ])
            
            batch_results.append(scenario_result)
            print(f"✅ Success: {len(smpl_results) if smpl_results else 0} people extracted")
            
        except Exception as e:
            print(f"❌ Error processing scenario {scenario['name']}: {str(e)}")
            batch_results.append({
                "scenario_id": i,
                "name": scenario['name'],
                "prompt": scenario['prompt'],
                "seed": scenario['seed'],
                "success": False,
                "error": str(e),
                "num_people": 0
            })
    
    # Save batch summary
    summary_path = "batch_output/batch_summary.json"
    os.makedirs("batch_output", exist_ok=True)
    
    batch_summary = {
        "total_scenarios": len(scenarios),
        "successful_scenarios": sum(1 for r in batch_results if r['success']),
        "total_people_extracted": total_people,
        "scenarios": batch_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print("\n" + "🎉 BATCH PROCESSING COMPLETE! 🎉".center(70))
    print("=" * 70)
    print(f"📊 Scenarios processed: {len(scenarios)}")
    print(f"✅ Successful scenarios: {batch_summary['successful_scenarios']}")
    print(f"🧑‍🤝‍🧑 Total people extracted: {total_people}")
    print(f"📁 Results saved in: batch_output/")
    print(f"📋 Summary saved in: {summary_path}")
    
    print("\n📈 Dataset ready for affordance research!")
    print("   • Use .obj files for 3D analysis")
    print("   • Use .json files for SMPL parameter studies") 
    print("   • Use .png renders for visual analysis")

if __name__ == '__main__':
    main()
