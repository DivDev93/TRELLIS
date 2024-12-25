from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import uuid
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Initialize the FastAPI app
app = FastAPI()

# Initialize the Trellis pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()  # Ensure this runs on a GPU-enabled environment

def process_image_to_3d(image_path: str, output_dir: str) -> str:
    """
    Takes an input image, runs it through the Trellis pipeline, and saves a .glb file.
    """
    # Open the image
    image = Image.open(image_path)

    # Run the Trellis pipeline
    outputs = pipeline.run(image)

    # Convert outputs to .glb format
    glb_path = os.path.join(output_dir, "output_model.glb")
    postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0]).export(glb_path)

    return glb_path

@app.post("/generate_model/")
async def generate_model(file: UploadFile = File(...)):
    """
    Endpoint for receiving an image, generating a 3D model, and returning the .glb file.
    """
    # Step 1: Save the uploaded file
    temp_dir = "/tmp/trellis_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Step 2: Create an output directory for the model
    output_dir = "/tmp/trellis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: Process the image into a 3D model
    try:
        glb_file_path = process_image_to_3d(image_path, output_dir)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # Step 4: Return the generated .glb file
    return FileResponse(glb_file_path, media_type="application/octet-stream", filename="output_model.glb")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
