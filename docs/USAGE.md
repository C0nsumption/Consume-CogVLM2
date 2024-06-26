# Usage Guide

This guide provides detailed instructions on how to use the CogVLM2 autocaptioning tools.

## Running the Script

You can run the script in three different modes: analyzing a single image or a directory of images, interacting with the model via a chat interface, or using a FastAPI application for HTTP requests.

### Single Image

To analyze a single image, run the following command:

```sh
python src/analyze.py path/to/image.jpg "Describe the image"
```

### Directory of Images

To analyze all images in a directory, run the following command:

```sh
python src/analyze.py path/to/directory "Describe the image"
```

### Saving Responses

To save the AI's responses to text files, add the `--save_response` flag:

```sh
python src/analyze.py path/to/image.jpg "Describe the image" --save_response
```

## Using the Chat Interface

You can interact with the CogVLM2 model via a CLI chat interface.

### Starting the Chat Interface

1. **Navigate to the Project Directory:**
    ```sh
    cd path/to/your/project
    ```

2. **Run the Chat Interface:**
    ```sh
    python src/chat.py
    ```

3. **Interacting with the Model:**

   - Provide the path to an image when prompted.
   - Enter your query to describe or ask questions about the image.
   - Type `clear` to clear the current session and provide a new image.

### Example Chat Session

```
Image path >>>>> path/to/image.jpg
Human: Describe the image
Assistant: The image shows a beautiful sunset over a mountain range with vibrant colors.
Human: What is the dominant color in the image?
Assistant: The dominant color in the image is orange, highlighting the sunset.
```

## Using the FastAPI Application

You can also use the CogVLM2 autocaptioning tools via a FastAPI application. This allows you to interact with the tools through HTTP requests.

### Starting the FastAPI Application

1. **Navigate to the Project Directory:**
    ```sh
    cd path/to/your/project
    ```

2. **Run the FastAPI Application:**
    ```sh
    python src/app.py
    ```

The FastAPI application will start and be available at `http://127.0.0.1:8000`.

### Analyzing an Image

To analyze an image, you can use a tool like `curl` or Postman to send a POST request to the `/analyze` endpoint.

#### Example `curl` Command:

```sh
curl -X POST "http://127.0.0.1:8000/analyze" \
-F "file=@path/to/your/image.jpg" \
-F "query=Describe the image" \
-F "max_new_tokens=2048" \
-F "save_response=false"
```

### Saving Responses

To save the AI's responses along with the uploaded image, set the `save_response` field to `true`:

```sh
curl -X POST "http://127.0.0.1:8000/analyze" \
-F "file=@path/to/your/image.jpg" \
-F "query=Describe the image" \
-F "max_new_tokens=2048" \
-F "save_response=true"
```

## Command Line Arguments

- `path`: Path to the image file or directory containing images.
- `query`: Query to ask about the image(s).
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 2048).
- `--save_response`: Save the response to a text file.

## Example Outputs

### Single Image

Command:

```sh
python src/analyze.py example.jpg "Describe the image"
```

Output:

```
==> example.jpg: The image captures a serene scene of a white crane, its wings spread wide in a display of majesty, standing on the shore of a tranquil lake.
```

### Directory of Images

Command:

```sh
python src/analyze.py images/ "Describe the images" --save_response
```

Output:

```
==> image1.jpg: The image captures a serene scene of a white crane, its wings spread wide in a display of majesty, standing on the shore of a tranquil lake.
Response saved to: images/image1.txt
==> image2.jpg: The image portrays a woman lying on a grassy field.
Response saved to: images/image2.txt
```

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure that all dependencies are installed correctly.
- Verify that the paths to the images are correct.
- Check the console output for any error messages and follow the suggestions provided.

For further assistance, feel free to open an issue on the [GitHub repository](https://github.com/your-repo-url/issues).

---

Happy autocaptioning!

---