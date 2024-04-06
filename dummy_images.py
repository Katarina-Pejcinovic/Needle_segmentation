from PIL import Image
import os

from PIL import Image
import os

def create_png_images(input_dir, output_dir):
    # Get list of jpg files in the input directory
    jpg_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, jpg_file in enumerate(jpg_files):
        # Generate PNG image name based on jpg file number
        png_name = f"{os.path.splitext(jpg_file)[0]}_mask.png"
        
        # Create a new blank image
        new_image = Image.new('L', (512, 512), color=0)
        
        # Save the image to the output directory
        new_image.save(os.path.join(output_dir, png_name))
        
        print(f"Created {png_name}")
    
    print("All PNG images created.")

# Example usage
input_directory = "/Users/katarinapejcinovic/Library/CloudStorage/OneDrive-UCLAITServices/Documents/college_stuff/Masters_classes/Advances_in_imaging/be224b-sp24-project/testImages/testImages"
output_directory = "output_images"
create_png_images(input_directory, output_directory)
