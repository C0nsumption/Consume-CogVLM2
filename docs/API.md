# API Documentation

This document provides detailed information about the classes and methods used in the CogVLM2 autocaptioning tools.

## ImageAnalyzer Class

A class to analyze images and generate descriptive captions using the CogVLM2 model.

### Methods

#### `__init__(self, model_path)`

Initializes the ImageAnalyzer class with the specified model path.

- **Parameters**:
  - `model_path` (str): Path to the model.

- **Example**:
  ```python
  analyzer = ImageAnalyzer("./cogvlm2-llama3-chat-19B-int4")
  ```

#### `apply_prompt_template(prompt)`

Applies a prompt template to the given prompt.

- **Parameters**:
  - `prompt` (str): The prompt to apply the template to.

- **Returns**: 
  - (str): The formatted prompt.

- **Example**:
  ```python
  formatted_prompt = ImageAnalyzer.apply_prompt_template("Describe the image")
  ```

#### `__call__(self, image, query, max_new_tokens=2048)`

Generates a prediction for the given image and query.

- **Parameters**:
  - `image` (PIL.Image): The image to analyze.
  - `query` (str): Query to ask about the image.
  - `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 2048).

- **Returns**:
  - (str): The prediction.

- **Example**:
  ```python
  prediction = analyzer(image, "Describe the image")
  ```

#### `analyze_image(self, image_path, query, max_new_tokens=2048, save_response=False)`

Analyzes a single image and prints the prediction.

- **Parameters**:
  - `image_path` (str): Path to the image file.
  - `query` (str): Query to ask about the image.
  - `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 2048).
  - `save_response` (bool, optional): Save the response to a text file (default: False).

- **Example**:
  ```python
  analyzer.analyze_image("path/to/image.jpg", "Describe the image", save_response=True)
  ```

#### `analyze_directory(self, directory_path, query, max_new_tokens=2048, save_response=False)`

Analyzes all images in a directory and prints the predictions.

- **Parameters**:
  - `directory_path` (str): Path to the directory containing images.
  - `query` (str): Query to ask about the images.
  - `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 2048).
  - `save_response` (bool, optional): Save the responses to text files (default: False).

- **Example**:
  ```python
  analyzer.analyze_directory("path/to/directory", "Describe the images", save_response=True)
  ```

---

For more information, please check the full usage guide and other documentation available in the repository.

---