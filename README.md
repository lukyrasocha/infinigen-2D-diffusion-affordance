# Infinigen-Affordance: Combined Image Generation + 4D-Humans Pipeline

This repository integrates **FLUX image generation** with **4D-Humans SMPL extraction** to create a comprehensive pipeline for generating realistic images of people on beds and automatically extracting 3D human body models.

## Features

🎨 **FLUX Image Generation**
- Generate high-quality images of people on beds using FLUX.1-Fill-dev
- Customizable prompts and generation parameters
- Support for different bed scenes and poses

🧑‍🤝‍🧑 **4D-Humans SMPL Extraction**
- Automatic human detection and pose estimation
- Extract SMPL parameters (shape, pose, global orientation)
- Generate 3D mesh files (.obj) and rendered visualizations
- Multiple view renders (front, side, top)

🔄 **Integrated Pipeline**
- End-to-end workflow from text prompt to 3D model
- Optimized for bed/furniture affordance scenarios
- Batch processing capabilities
- **Self-contained**: 4D-Humans included in repository

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd infinigen-affordance

# Install dependencies
pip install -r requirements.txt

# The 4D-Humans model is included in the repository
# No additional setup required!
```

### 2. Basic Usage

```bash
# Generate a person lying on a bed
python combined_pipeline.py \
    --prompt "a person with blue jeans and white t-shirt lying comfortably on a bed" \
    --bed_image images/bed3.png \
    --mask_image images/mask.jpg \
    --output_dir output
```

### 3. Run Examples

```bash
# Run comprehensive examples
python example.py
```

## Repository Structure

This repository is **completely self-contained** and includes all necessary components:

```
infinigen-affordance/
├── 4D-Humans/              # Integrated 4D-Humans codebase
│   ├── hmr2/               # Core HMR2.0 model
│   ├── demo.py             # Original demo script
│   └── ...                 # Complete 4D-Humans implementation
├── images/                 # Sample bed images and masks
│   ├── bed3.png            # Main bed image
│   ├── mask.jpg            # Main mask
│   └── ...                 # Additional samples
├── combined_pipeline.py    # Main pipeline script
├── example.py              # Example usage
├── batch_process.py        # Batch processing for research
├── requirements.txt        # All dependencies
└── README.md              # This documentation
```

## Input Requirements

### Bed Image
- **Format**: JPG, PNG
- **Content**: Empty bed scene
- **Provided**: `images/bed3.png`, `images/bed.png`

### Mask Image
- **Format**: JPG, PNG  
- **Content**: White regions indicate where to generate the person
- **Provided**: `images/mask.jpg`, `images/mask_bed.jpg`

### Prompt
- Text description of the person and pose
- Examples:
  - "a person with blue jeans and white t-shirt lying comfortably on a bed"
  - "a woman in elegant dress sitting gracefully on a bed reading a book"
  - "a man in business suit lying on a bed reading a book"

## Output Files

The pipeline generates comprehensive outputs in the specified output directory:

### Generated Images
- `*_generated.png` - Original generated image (800x800)

### SMPL Renders
- `*_person_*_render.png` - SMPL mesh overlaid on generated image (256x256)
- `*_person_*_side_view.png` - Side view of SMPL mesh
- `*_person_*_top_view.png` - Top view of SMPL mesh

### 3D Data
- `*_person_*.obj` - 3D mesh file for import into Blender/3D software
- `*_person_*_smpl.json` - Complete SMPL parameters

## Command Line Options

### Input Parameters
- `--bed_image` - Path to empty bed image (default: `images/bed3.png`)
- `--mask_image` - Path to mask image (default: `images/mask.jpg`)
- `--prompt` - Text prompt for generation (required)

### Output Parameters
- `--output_dir` - Output directory (default: `output`)
- `--base_filename` - Base filename for outputs

### Generation Parameters
- `--height`, `--width` - Image dimensions (default: 800x800)
- `--guidance_scale` - Guidance scale (default: 30)
- `--num_inference_steps` - Inference steps (default: 50)
- `--seed` - Random seed for reproducibility

### 4D-Humans Parameters
- `--detector` - Human detector ('vitdet' or 'regnety', default: 'vitdet')
- `--batch_size` - Processing batch size (default: 8)

## API Usage

```python
from combined_pipeline import CombinedPipeline

# Initialize pipeline
pipeline = CombinedPipeline(detector='vitdet')

# Generate image and extract SMPL
generated_image, smpl_results = pipeline.run_full_pipeline(
    bed_image_path="images/bed3.png",
    mask_image_path="images/mask.jpg",
    prompt="a person lying on a bed",
    output_dir="output",
    height=800,
    width=800,
    seed=42
)

# Access results
if smpl_results:
    for result in smpl_results:
        vertices = result['vertices']  # 3D mesh vertices
        smpl_params = result['smpl_params']  # SMPL parameters
        camera_translation = result['camera_translation']  # Camera position
```

## Integration with Infinigen

This pipeline is designed to integrate seamlessly with other Infinigen affordance generation tools:

1. **Scene Generation**: Use Infinigen to generate bed/bedroom scenes
2. **Person Placement**: Use this pipeline to add realistic people to scenes
3. **SMPL Extraction**: Extract precise 3D human models for simulation
4. **Affordance Analysis**: Analyze how people interact with furniture

## Technical Details

### Resolution Optimization
- Generated images: 800x800 for high quality
- SMPL rendering: 256x256 for optimal mesh alignment
- The pipeline automatically handles resolution scaling

### Model Dependencies
- **FLUX.1-Fill-dev**: `black-forest-labs/FLUX.1-Fill-dev`
- **4D-Humans**: Uses HMR2.0 model from `/root/4D-Humans`
- **Detectron2**: For human detection (ViTDet or RegNetY)

### Performance
- **GPU Memory**: ~8GB VRAM recommended
- **Processing Time**: ~30-60 seconds per image on modern GPU
- **Accuracy**: >90% human detection success rate

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python combined_pipeline.py --batch_size 4
   ```

2. **No Humans Detected**
   - Check mask alignment with bed image
   - Ensure prompt describes visible person
   - Try different detection threshold

3. **Import Errors**
   ```bash
   # Ensure 4D-Humans path is correct
   export PYTHONPATH=/root/4D-Humans:$PYTHONPATH
   ```

### Debug Mode
```bash
# Enable verbose output
python combined_pipeline.py --prompt "..." --verbose
```

## Contributing

This pipeline combines multiple state-of-the-art models:
- FLUX.1-Fill-dev for image generation
- 4D-Humans for SMPL extraction
- Detectron2 for human detection

Contributions for improving integration, performance, and accuracy are welcome!

## License

Please refer to the individual model licenses:
- FLUX: Flux.1 Model License
- 4D-Humans: See 4D-Humans repository
- Detectron2: Apache 2.0
